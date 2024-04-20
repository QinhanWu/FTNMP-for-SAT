import torch as tc
import networkx as nx
import copy 
import opt_einsum as oe
import time
import subprocess
import json

def generate_graph(clauses,fac_list):
    edges = []
    max_node = 0
    for clause_id in range(len(clauses)):
        clause = clauses[clause_id]
        qubit2 = fac_list[clause_id]
        max_node = max(max_node, qubit2)
        for qubit1_id in range(len(clause)):
            qubit1 = clause[qubit1_id]
            max_node = max(max_node, qubit1)
            edges.append((qubit1, qubit2))
    return edges, max_node + 1

def G_generator_fac(clauses):
    G = nx.Graph()    #前n个是qubit，后面是clause
    if len(clauses)==0:
        max_item = 0
    else:
        max_item = max(max(row) for row in clauses if len(row)!=0)
        for clause_id in range(len(clauses)):
            clause = clauses[clause_id]
            if len(clause)!=0:
                qubit2 = max_item + clause_id + 1
                for qubit1_id in range(len(clause)):
                    qubit1 = clause[qubit1_id]
                    G.add_edge(qubit1,qubit2)
    return G,max_item

def TNBP_3_sat(G_fac,Nv,boundaries,clauses,tendencies,max_item,device):
    converged = True
    n = G_fac.number_of_nodes()
    #print(n)
    step_limit = 1000
    epsilon = 1e-6
    difference_max = tc.tensor([0.0]).to(device)
    damping_factor = 0
    egdelist2  = [sorted(edge) for edge in G_fac.edges()]
    cavity = (tc.ones(size=(n,n,4))/2).to(device)
    paths = [[[]for w in range(n)] for d in range(n)]
    t1 = time.time()
    for step in range(step_limit):
        first_center_node = -1
        for center_node in range(n):
            first_boundary_node = -1
            neighborhood = Nv[center_node]
            if (len(boundaries[center_node]) != 0 and first_center_node == -1):
                first_center_node = center_node
            for node in neighborhood:
                if node in boundaries[center_node]:
                    if first_boundary_node == -1:
                        first_boundary_node = node
                    Ne_cavity = list(set(Nv[node]).difference(Nv[center_node]))
                    if step > 0 :
                        New_cavity_vec = local_contraction(Ne_cavity,clauses,tendencies,cavity,node,max_item,egdelist2,paths[node][center_node],device)
                    else:
                        New_cavity_vec,my_contract = local_contraction_begin(G_fac,Ne_cavity,clauses,tendencies,cavity,node,max_item,egdelist2,device)
                        paths[node][center_node]=my_contract    
                    temp = damping_factor * cavity[node][center_node] + (1 - damping_factor) * New_cavity_vec
                    temp /= tc.sum(temp)
                    difference = max(abs(temp[0]-cavity[node][center_node][0]), 
                                     abs(temp[1]-cavity[node][center_node][1]),
                                     abs(temp[2]-cavity[node][center_node][2]),
                                     abs(temp[3]-cavity[node][center_node][3]))
                    cavity[node][center_node] = temp
                else:
                    cavity[node][center_node] = tc.tensor([1.0,1.0,0,0]).to(device)
                    cavity[node][center_node] = cavity[node][center_node]/tc.sum(cavity[node][center_node])
                    difference =tc.tensor(0.0).to(device)   
                if center_node == first_center_node and node == first_boundary_node:
                    difference_max = difference
                elif difference > difference_max:
                    difference_max = difference   
        t2 = time.time()
        print("iteration step:",step+1,",   difference:",difference_max.item(),'    time:',t2-t1)
        if difference_max <= epsilon :
            break   
    if step==step_limit-1:
        print('unconverged') 
        converged = False  
    return cavity ,converged,egdelist2

def local_contraction_sat_fac(G_fac,Ne_cavity,clauses,tendencies,cavity,node,max_item,egdelist2,device):
    if node>max_item:
        tensors = []
        eqin = ''
        eqout = ''
        node_nei = list(G_fac.neighbors(node))
        out_legs = sorted([item for item in node_nei if item in Ne_cavity])
        for out_leg in out_legs:
            eqout += oe.get_symbol(out_leg)
    else:
        tensors = [tc.tensor([0.5,0.5]).to(device)]
        eqin = ''
        eqin += oe.get_symbol(node)
        eqin +=','
        eqout = ''
        eqout += oe.get_symbol(node)
    for noden in Ne_cavity:
        if noden>max_item:
            clause = clauses[noden - max_item-1]
            tensor = (tc.ones(size=[2]*len(clause))*(1.0)).to(device)
            zero_pos = [[int(-0.5*i+0.5)] for i in tendencies[noden - max_item-1]]
            tensor[zero_pos] = 0.0
            tensors.append(tensor)
            nodes_subset =sorted([item for item in clause if item not in Ne_cavity and item != node])
            nodes_subset_copy = copy.deepcopy(nodes_subset)
            for node_in_clause in clause:
                if node_in_clause  in nodes_subset_copy:
                    Q = set(G_fac.neighbors(node_in_clause))
                    R = set(Ne_cavity)
                    if len(Q.intersection(R))>1:
                        symbol_id = egdelist2.index([node_in_clause,noden])
                        eqin +=oe.get_symbol(symbol_id+max_item+1)
                        node_in_clause_id = nodes_subset.index(node_in_clause)
                        nodes_subset[node_in_clause_id]=symbol_id+max_item+1
                    else:
                        eqin += oe.get_symbol(node_in_clause)
                else:
                    eqin += oe.get_symbol(node_in_clause)
            eqin += ","
            if len(nodes_subset)==2:
                tensor_pre = cavity[noden][node]
                tensor=(tensor_pre.reshape(2,2)).to(device)
                tensors.append(tensor)
                for nodet in nodes_subset:
                    eqin +=oe.get_symbol(nodet)
                eqin +=','
            if len(nodes_subset)==1:
                tensor = (tc.tensor([cavity[noden][node][0],cavity[noden][node][1]])).to(device)
                tensors.append(tensor)
                eqin += oe.get_symbol(nodes_subset[0])
                eqin += ","
            

        else:
            tensor = tc.tensor([cavity[noden][node][0],cavity[noden][node][1]]).to(device)
            tensors.append(tensor)
            eqin += oe.get_symbol(noden)
            eqin += ","
    eqin = eqin.rstrip(',')
    eqin += '->'
    eq = eqin + eqout
    z = oe.contract(eq, *tensors, optimize=True)
    if len(z.size())== 1:
       z = (tc.tensor([z[0],z[1],0,0])).to(device)
    if len(z.size())== 2:
       z=(z.reshape(4)).to(device) 
    con = (tc.zeros(4)).to(device)
    if tc.equal(z,con)==True:
        z =(tc.tensor([0.5,0.5,0.5,0.5])).to(device)
    return z

def local_contraction(Ne_cavity,clauses,tendencies,cavity,node,max_item,egdelist2,my_contract,device):
    if node>max_item:
        tensors = []
    else:
        tensors = [tc.tensor([0.5,0.5]).to(device)]
    for noden in Ne_cavity:
        if noden>max_item:
            clause = clauses[noden - max_item-1]
            tensor = (tc.ones(size=[2]*len(clause))*(1.0)).to(device)
            zero_pos = [[int(-0.5*i+0.5)] for i in tendencies[noden - max_item-1]]
            tensor[zero_pos] = 0.0
            tensors.append(tensor)
            nodes_subset =sorted([item for item in clause if item not in Ne_cavity and item != node])
            if len(nodes_subset)==2:
                tensor_pre = cavity[noden][node]
                tensor=(tensor_pre.reshape(2,2)).to(device)
                tensors.append(tensor)
            if len(nodes_subset)==1:
                tensor = (tc.tensor([cavity[noden][node][0],cavity[noden][node][1]])).to(device)
                tensors.append(tensor)
        else:
            tensor = tc.tensor([cavity[noden][node][0],cavity[noden][node][1]]).to(device)
            tensors.append(tensor)
    z = my_contract(*tensors)
    if len(z.size())== 1:
       z = (tc.tensor([z[0],z[1],0,0])).to(device)
    if len(z.size())== 2:
       z=(z.reshape(4)).to(device) 
    con = (tc.zeros(4)).to(device)
    if tc.equal(z,con)==True:
        z =(tc.tensor([0.5,0.5,0.5,0.5])).to(device)
    return z

def local_contraction_begin(G_fac,Ne_cavity,clauses,tendencies,cavity,node,max_item,egdelist2,device):
    path = []
    if node>max_item:
        tensors = []
        eqin = ''
        eqout = ''
        node_nei = list(G_fac.neighbors(node))
        out_legs = sorted([item for item in node_nei if item in Ne_cavity])
        for out_leg in out_legs:
            eqout += oe.get_symbol(out_leg)
    else:
        tensors = [tc.tensor([0.5,0.5]).to(device)]
        eqin = ''
        eqin += oe.get_symbol(node)
        eqin +=','
        eqout = ''
        eqout += oe.get_symbol(node)
        path.append((2,))
        #print('eqout',eqout)
    for noden in Ne_cavity:
        if noden>max_item:
            clause = clauses[noden - max_item-1]
            tensor = (tc.ones(size=[2]*len(clause))*(1.0)).to(device)
            zero_pos = [[int(-0.5*i+0.5)] for i in tendencies[noden - max_item-1]]
            tensor[zero_pos] = 0.0
            tensors.append(tensor)
            path.append((2,)*len(clause))
            nodes_subset =sorted([item for item in clause if item not in Ne_cavity and item != node])
            nodes_subset_copy = copy.deepcopy(nodes_subset)
            for node_in_clause in clause:
                if node_in_clause  in nodes_subset_copy:
                    Q = set(G_fac.neighbors(node_in_clause))
                    R = set(Ne_cavity)
                    if len(Q.intersection(R))>1:
                        symbol_id = egdelist2.index([node_in_clause,noden])
                        eqin +=oe.get_symbol(symbol_id+max_item+1)
                        node_in_clause_id = nodes_subset.index(node_in_clause)
                        nodes_subset[node_in_clause_id]=symbol_id+max_item+1
                    else:
                        eqin += oe.get_symbol(node_in_clause)
                else:
                    eqin += oe.get_symbol(node_in_clause)
            eqin += ","
            if len(nodes_subset)==2:
                tensor_pre = cavity[noden][node]
                tensor=(tensor_pre.reshape(2,2)).to(device)
                tensors.append(tensor)
                for nodet in nodes_subset:#sorted(nodes_subset):
                    eqin +=oe.get_symbol(nodet)
                eqin +=','
                path.append((2,2))
            if len(nodes_subset)==1:
                tensor = (tc.tensor([cavity[noden][node][0],cavity[noden][node][1]])).to(device)
                tensors.append(tensor)
                eqin += oe.get_symbol(nodes_subset[0])
                eqin += ","
                path.append((2,))
        else:
            tensor = tc.tensor([cavity[noden][node][0],cavity[noden][node][1]]).to(device)
            tensors.append(tensor)
            eqin += oe.get_symbol(noden)
            eqin += ","
            path.append((2,))
    eqin = eqin.rstrip(',')
    eqin += '->'
    eq = eqin + eqout
    my_con = oe.contract_expression(eq,*path)
    z = my_con(*tensors)
    #z = z/tc.norm(z)#!!!!!
    if len(z.size())== 1:
       z = (tc.tensor([z[0],z[1],0,0])).to(device)
    if len(z.size())== 2:
       z=(z.reshape(4)).to(device) 
    con = (tc.zeros(4)).to(device)
    if tc.equal(z,con)==True:
        z =(tc.tensor([0.5,0.5,0.5,0.5])).to(device)
    
    return z,my_con

def marginal_TNBP(G_fac,nodes,egdelist2,Nv,boundaries,clauses,tendencies,cavity,max_item,device):
    margnals_TNBP =[]
    for tar_node in nodes:
        #print(tar_node)
        #print(Nv[tar_node])
        #print(boundaries[tar_node])
        tensors = []
        eqin = ''
        eqout = ''
        for noden in Nv[tar_node]:
            if noden>max_item and noden - max_item-1<len(clauses):
                clause = clauses[noden - max_item-1]
                tensor = (tc.ones(size=[2]*len(clause))*(1.0)).to(device)
                zero_pos = [[int(-0.5*i+0.5)] for i in tendencies[noden - max_item-1]]
                tensor[zero_pos] = 0.0
                tensors.append(tensor)
                for node_in_clause in clause:
                    Q = set(G_fac.neighbors(node_in_clause))
                    R = set(Nv[tar_node])
                    if len(Q.intersection(R))>1 and noden in boundaries[tar_node] and node_in_clause not in Nv[tar_node]:
                        symbol_id = egdelist2.index([node_in_clause,noden])
                        eqin +=oe.get_symbol(symbol_id+max_item+1)
                    else:
                        eqin += oe.get_symbol(node_in_clause)
                eqin += ","
            if noden<=max_item and noden!= tar_node:
                #print(noden)
                tensor = (tc.tensor([0.5,0.5])).to(device)
                tensors.append(tensor)
                eqin +=oe.get_symbol(noden)
                eqin += ","   
            if noden == tar_node:
                tensor = (tc.tensor([0.5,0.5])).to(device)
                eqin += oe.get_symbol(noden)
                eqout += oe.get_symbol(noden)
                eqin += ","   
                tensors.append(tensor)
        for nodey in boundaries[tar_node]:
            nodey_nei = list(G_fac.neighbors(nodey))
            message = sorted([item for item in nodey_nei if item not in Nv[tar_node]],reverse=False)
            tensor_pre = cavity[nodey][tar_node]
            if len(message)==1:
               tensor =(tc.tensor([tensor_pre[0],tensor_pre[1]])).to(device)
               tensors.append(tensor)
               if message[0]<=max_item:
                    Q = set(G_fac.neighbors(message[0]))
                    R = set(Nv[tar_node])
                    if len(Q.intersection(R))>1:
                        symbol_id = egdelist2.index([message[0],nodey])
                        eqin +=oe.get_symbol(symbol_id+max_item+1)
                    else:
                        eqin += oe.get_symbol((message[0]))
               else:
                   eqin += oe.get_symbol(nodey)
               eqin += ","
            #if len(message)==2:
            else:
                if nodey>max_item:
                    tensor = (tensor_pre.reshape(2,2)).to(device)
                    tensors.append(tensor)
                    for message_id in range(len(message)):
                        Q = set(G_fac.neighbors(message[message_id]))
                        R = set(Nv[tar_node])
                        if len(Q.intersection(R))>1:
                            symbol_id = egdelist2.index([message[message_id],nodey])
                            eqin +=oe.get_symbol(symbol_id+max_item+1)
                        else:
                            eqin += oe.get_symbol((message[message_id]))
                    eqin += ","
                else:
                    tensor = (tc.tensor([tensor_pre[0],tensor_pre[1]])).to(device)
                    tensors.append(tensor)
                    eqin += oe.get_symbol(nodey)
                    eqin += ","
        eqin = eqin.rstrip(',') 
        eqin += '->'
        eq = eqin +eqout
        if len(tensors)!=0:
            marginal = oe.contract(eq, *tensors, optimize=True)
            marginal = (marginal/(marginal[0]+marginal[1])).to(device)
            margnals_TNBP.append(marginal)
        else:
            marginal = (tc.tensor([0.5,0.5])).to(device)
            margnals_TNBP.append(marginal)
    return margnals_TNBP

