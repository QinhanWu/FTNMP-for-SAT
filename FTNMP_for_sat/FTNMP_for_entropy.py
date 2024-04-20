import graph_generate as gg
import TNBP
import torch as tc
import opt_einsum as oe
import copy
import BP as bpf
import constract_exact as ce
import time



def local_constraction_for_site(tar_node,target_fac,G_fac,Nv,boundaries,clauses,tendencies,cavity,max_item,device):
    egdelist2 = [sorted(edge) for edge in G_fac.edges()]
    tensors = []
    eqin = ''
    eqout = ''
    for noden in Nv[tar_node]:
        if noden > max_item and noden - max_item-1<len(clauses) and noden!=target_fac:
            clause = clauses[noden - max_item-1]
            tensor = (tc.ones(size=[2]*len(clause))*(1.0)).to(device)
            zero_pos = [[int(-0.5*i+0.5)] for i in tendencies[noden - max_item-1]]
            tensor[zero_pos] = 0.0+1e-40
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
    else:
        marginal = (tc.tensor([0.5,0.5])).to(device)
    return marginal

def local_constraction_for_divide(tar_node,G_fac,Nv,boundaries,clauses,tendencies,cavity,max_item,device):
    egdelist2 = [sorted(edge) for edge in G_fac.edges()]
    tensors = []
    eqin = ''
    eqout = ''
    for noden in Nv[tar_node]:
        if noden > max_item and noden - max_item-1<len(clauses):
            clause = clauses[noden - max_item-1]
            tensor = (tc.ones(size=[2]*len(clause))*(1.0)).to(device)
            zero_pos = [[int(-0.5*i+0.5)] for i in tendencies[noden - max_item-1]]
            tensor[zero_pos] = 0.0+1e-10
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
        if noden<=max_item :
            eqout +=oe.get_symbol(noden)
    for nodey in boundaries[tar_node]:
        nodey_nei = list(G_fac.neighbors(nodey))
        message = sorted([item for item in nodey_nei if item not in Nv[tar_node]],reverse=False)
        if nodey <= max_item:
            eqout +=oe.get_symbol(nodey)
        else:
            for c in message:
                eqout +=oe.get_symbol(c)
        #print('!','target_node',tar_node,'nodey',nodey,set(nodey_nei),set(message))
        #cavity_id = devide[tar_node].index(list(set(nodey_nei).difference(set(message)))[0])
        #tensor_pre = cavity[nodey][devide[tar_node][cavity_id]]
        cavity_id = Nv[tar_node].index(list(set(nodey_nei).difference(set(message)))[0])
        tensor_pre = cavity[nodey][Nv[tar_node][cavity_id]]
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
    #print(eq)
    #print(tensors,file=file)
    if len(tensors)!=0:
        marginal = oe.contract(eq, *tensors, optimize=True)
        marginal = (marginal/tc.sum(marginal)).to(device)
        #print(marginal,file = file)
    else:
        marginal = (tc.tensor([0.5,0.5])).to(device)
        #print(marginal,file = file)
    return marginal

def marginal_TNBP_entropy(target_fac,G_fac,nodes,Nv,boundaries,clauses,tendencies,cavity,max_item,device):
    margnals_TNBP =[]
    for tar_node in nodes:
        marginal = local_constraction_for_site(tar_node,target_fac,G_fac,Nv,boundaries,clauses,tendencies,cavity,max_item,device)
        margnals_TNBP.append(marginal)
    return margnals_TNBP

def cluster_marginals(G_fac,clauses,tendencies,devide,devide_bound,max_item,cavity,device):
    marginal_devides = []
    for devide_id in range(len(devide)):
        marginal_devide = local_constraction_for_divide(devide_id,G_fac,devide,devide_bound,clauses,tendencies,cavity,max_item,device)
        marginal_devides.append(marginal_devide)


    return marginal_devides

def divide_change(devide,devide_boud,devide_node,add_clauses,clauses,R):
    devide_copy = copy.deepcopy(devide)
    devide_boud_copy = copy.deepcopy(devide_boud)
    devide_node_copy = copy.deepcopy(devide_node)
    max_item = max(max(clauses))
    for d_id in range(len(devide)):

        if len(devide[d_id]) > 2*R :
            devide_copy.remove(devide[d_id])
            devide_boud_copy.remove(devide_boud[d_id])
            for node in devide[d_id]:
                devide_node_copy.remove(node)
                for clause in add_clauses:
                    clause_id = clauses.index(clause)+max_item+1
                    if node in clause and  [clause_id] not in devide_copy:
                        devide_copy.append([clause_id])
                        devide_boud_copy.append(clause)
            
    return devide_copy,devide_boud_copy,devide_node_copy

def marginal_var_BP_entropy(G_edges,nodes,clauses,tendencies,result_BP,max_item):
    m = tc.sum(result_BP)
    if tc.isnan(m).any()==True:
        #print('no solution')
        return('no solution')
    else:
        Ws= []
        for node in nodes:
            k =0
            for clauseo in clauses:
                if node in clauseo:
                    k=k+1
            if k!=0:
                neibor_of_node = bpf.neiborhood_sat(node,clauses,max_item)#全是因素节点
            else:
                neibor_of_node=[] 
            V_up = []
            V_down = []
            for nodec in neibor_of_node:
                clause = clauses[nodec - max_item-1]
                tendencyc_id = clause.index(node)
                sign = tendencies[nodec - max_item-1][tendencyc_id]
                if sign == 1:
                    V_up.append(nodec)
                else:
                    V_down.append(nodec)
            pi_up = 1
            for nodeq in V_up:
                messageq_id = G_edges.index([node,nodeq])
                pi_up = pi_up*(1-result_BP[messageq_id])
            pi_down = 1
            for nodeg in V_down:
                messageg_id = G_edges.index([node,nodeg])
                pi_down = pi_down*(1-result_BP[messageg_id])
            PI_UP = (pi_up)
            PI_DOWN = (pi_down)
            #W = tc.zeros(2)
            if (PI_UP+PI_DOWN)==0:
                #print('no solution')
                Ws.append(tc.tensor([-404,-404]))#这里表示有些点是无解的
            else:
                WI_up = PI_UP#/(PI_UP+PI_DOWN)
                WI_down=PI_DOWN#/(PI_DOWN+PI_UP)
                W =PI_UP+PI_DOWN
                #W[0]=WI_up
                #W[1]=WI_down
                Ws.append(W)
        #print(Ws)
        return Ws

def marginal_fac_BP_entropy(G_edges,nodes,clauses,tendencies,result_BP,max_item):
    m = tc.sum(result_BP)
    if tc.isnan(m).any()==True:
        #print('no solution')
        return('no solution')
    else:
        Ws= []
        
        for cla_id in range(len(clauses)):
            t1 = 1
            t2 = 1 
            clause = clauses[cla_id]
            for var_id in range(len(clause)):
                Vs = []
                Vu = []
                u_term = 1
                s_term = 1
                var_node = clause[var_id]
                nei_var_nodes = bpf.neiborhood_sat(var_node,clauses,max_item)
                tendency = tendencies[cla_id][var_id]
                for nei_var_node in nei_var_nodes:
                    clause_id = nei_var_node - max_item -1
                    var_node_id = clauses[clause_id].index(var_node)
                    if tendencies[clause_id][var_node_id]*tendency ==1 and nei_var_node != max_item+1+cla_id:
                        #Vs.append(var_node)
                        message_id = G_edges.index([var_node,nei_var_node])
                        s_term = s_term*(1-result_BP[message_id])
                    if tendencies[clause_id][var_node_id]*tendency ==-1 and nei_var_node != max_item+1+cla_id:
                        #Vu.append(var_node)
                        #message_id = G_edges.index([clause[id],max_item+1+clause_id])
                        message_id = G_edges.index([var_node,nei_var_node])
                        u_term = u_term*(1-result_BP[message_id])
                t1 = t1*(u_term+s_term)
                t2 = t2*s_term
            Ws.append(t1-t2)
        return Ws

    


device = 'cpu'
t = time.time()
turns = 10
n_long = 10
n_trangel = 10
R = 4
clauses,tendencies,n_trange_list,add_clauses,devide,devide_bound,devide_node = gg.triangle_long_graph_free(turns,n_trangel,n_long)
G_fac,max_item = gg.G_generator_fac(clauses)
nodes = list(range(max_item+1))
#Nv,boundaries = gg.neighbor(max_item,clauses,R)
Nv,boundaries = gg.boundaries_generation(G_fac,clauses,R)
cavity,converged,edgelist2 = TNBP.TNBP_3_sat(G_fac,Nv,boundaries,clauses,tendencies,max_item,device)
marginals_var=TNBP.marginal_TNBP(G_fac,nodes,edgelist2,Nv,boundaries,clauses,tendencies,cavity,max_item,device)
marginals_fac = cluster_marginals(G_fac,clauses,tendencies,devide,devide_bound,max_item,cavity,device)
entropy = 0
for var_node in range(len(marginals_var)):
    if var_node not in devide_node:
        entropy = entropy + (1-G_fac.degree(var_node))*tc.dot(marginals_var[var_node],tc.log(marginals_var[var_node]))
for de_node in range(len(marginals_fac)):
    p = marginals_fac[de_node].numel()
    x = marginals_fac[de_node].reshape(p)
    entropy = entropy + tc.dot(x,tc.log(x))
entropy = -entropy
print(entropy)

configrite_number,norm = ce.local_contraction_sat_fac(G_fac,clauses,tendencies,max_item,device)
entropy_exact = tc.log(configrite_number)+norm
print(entropy_exact)

print('error:',abs(entropy_exact-entropy)/entropy_exact)



