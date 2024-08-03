import torch as tc
import numpy as np
import copy
import itertools as it
import time


def G_generate_bpf(clauses):
    
    G_edge = []
    clauses_1d = list(it.chain.from_iterable(clauses))
    node_list = np.unique(clauses_1d)
    if len(node_list)==0:
        max_item=0
    else:
        max_item = max(node_list)
        for clause_id in range(len(clauses)):
            clause = clauses[clause_id]
            if len(clause)!=0:
                qubit2 = max_item + clause_id + 1
                for qubit1_id in range(len(clause)):
                    qubit1 = clause[qubit1_id]
                    G_edge.append([qubit1,qubit2])
    G_edge=sorted(G_edge)
    G = np.array(G_edge)
    slice = []
    for i in node_list:
        X = np.argwhere(G==i)
        slice.append(list(X[:,0]))
    return G_edge,max_item,node_list,slice

def sat_clause(NODE_connect,configurat,clauses,tendencies):
    Es=[]
    for clause_id in range(len(clauses)):
        clause = clauses[clause_id]
        Ez = 1
        for node_id in range(len(clause)):
            point = clause[node_id]
            if point in NODE_connect:
                configurat_id = NODE_connect.index(point)
                Ei = (1-configurat[configurat_id]*tendencies[clause_id][node_id])/2
                Ez = Ez*Ei
        Es.append(Ez)
    E = sum(Es)
    return E,Es

def neiborhood_sat(centernode,clauses,max_item):
    neiborhood=[]
    if centernode>max_item:
        neiborhood = clauses[centernode-max_item-1]
    else:
        for clause_id in range(len(clauses)):
            clause = clauses[clause_id]
            if centernode in clause:
                neiborhood.append(max_item + clause_id + 1)
    return neiborhood

def information_of_fac(G_edges,clauses,tendencies,max_item):
    information = tc.zeros(len(clauses),6,len(G_edges))
    for fac_id in range(len(clauses)):
        fac_node = fac_id + max_item + 1  
        neibor_of_fac = clauses[fac_id]    
        for var_id in range(len(neibor_of_fac)):   
            var_node = neibor_of_fac[var_id]    #!!!
            for edge in G_edges:
                if edge[0] in neibor_of_fac and edge[0] != var_node and edge[1] != fac_node:
                    pos = clauses[edge[1]-max_item-1].index(edge[0])
                    sign_var_to_fac = tendencies[edge[1]-max_item-1][pos]
                    vare_id = clauses[fac_id].index(edge[0])
                    sign_fac_to_var = tendencies[fac_id][vare_id]
                    row_id = int(np.heaviside(sign_var_to_fac*sign_fac_to_var,0))    
                    edge_id = G_edges.index(edge)
                    information[fac_id][2*var_id+row_id][edge_id]=1              
    return information

def belief_propagation_fast(G_edges,node_list,slice,clauses,tendencies,max_item):
    interation = 1000
    epsilon = 1e-3
    damping = 0.5
    message_fac_to_var = tc.ones(len(G_edges))
    message_fac_to_var=message_fac_to_var/2
    information = information_of_fac(G_edges,clauses,tendencies,max_item)
    t1 = time.time()
    for t in range(interation):
        message_fac_to_var_copy = copy.deepcopy(message_fac_to_var)
        for edge in G_edges:
            message_fac_to_var_ln = tc.log(1-message_fac_to_var+1e-30)
            fac_id = edge[1] - max_item - 1
            var_u_id = clauses[fac_id].index(edge[0])
            clausey = copy.deepcopy(clauses[fac_id])
            clausey.remove(edge[0])
         
            message_new = 1
            for nei_node in clausey:    
                slice_id = list(node_list).index(nei_node)
                sliced = slice[slice_id]
                message_middle  = tc.exp(information[fac_id,2*var_u_id:2*var_u_id+2,min(sliced):max(sliced)+1]@message_fac_to_var_ln[min(sliced):max(sliced)+1].t())
                pi_u = message_middle[1]
                pi_s = message_middle[0]
                message_new = message_new*(pi_u/(pi_u+pi_s+1e-30))
            message_fac_to_var_id = G_edges.index(edge)
            message_fac_to_var[message_fac_to_var_id]=(1-damping)*message_new+damping*message_fac_to_var[message_fac_to_var_id]
        differences = message_fac_to_var_copy - message_fac_to_var
        if len(differences)!=0:
            difference = max(abs(differences))
            t2 = time.time()
            print('step:',t+1,'    difference:',difference,'    time:',t2-t1)
        else:
            break
        if difference<epsilon:
            break
    success_or_not = 'converged'
    if t==interation-1:
        success_or_not = 'unconverged'
        print('unconverged')
    return message_fac_to_var,success_or_not,t+1
'''
def marginal_BP(G_edges,nodes,clauses,tendencies,result_BP,max_item):
    m = tc.sum(result_BP)
    if tc.isnan(m).any()==True:
        return('no solution')
    else:
        Ws= []
        for node in nodes:
            k =0
            for clauseo in clauses:
                if node in clauseo:
                    k=k+1
            if k!=0:
                neibor_of_node = neiborhood_sat(node,clauses,max_item)
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
            W = tc.zeros(2)
            if (PI_UP+PI_DOWN)==0:
                Ws.append(tc.tensor([-404,-404]))
            else:
                WI_up = PI_UP/(PI_UP+PI_DOWN)
                WI_down=PI_DOWN/(PI_DOWN+PI_UP)
                W[0]=WI_up
                W[1]=WI_down
                Ws.append(W)
        return Ws
'''

