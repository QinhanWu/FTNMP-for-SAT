import torch as tc
import networkx as nx
import opt_einsum as oe
import ftnmp.graph_generate as gg
import ftnmp.BP_fast as bpf




#BP
def marginal_var_BP_entropy(G_edges,nodes,clauses,tendencies,result_BP,max_item):
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
                neibor_of_node = bpf.neiborhood_sat(node,clauses,max_item)
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
            pi_up = tc.tensor(1)
            for nodeq in V_up:
                messageq_id = G_edges.index([node,nodeq])
                pi_up = pi_up*(1-result_BP[messageq_id])
            pi_down = tc.tensor(1)
            for nodeg in V_down:
                messageg_id = G_edges.index([node,nodeg])
                pi_down = pi_down*(1-result_BP[messageg_id])
            PI_UP = (pi_up)
            PI_DOWN = (pi_down)
            if (PI_UP+PI_DOWN)==0:
                Ws.append(tc.tensor([-404,-404]))
            else:
                W =PI_UP+PI_DOWN
                Ws.append(W)
        return Ws
    
def marginal_fac_BP_entropy(G_edges,nodes,clauses,tendencies,result_BP,max_item):
    m = tc.sum(result_BP)
    if tc.isnan(m).any()==True:
        return('no solution')
    else:
        Ws= []
        
        for cla_id in range(len(clauses)):
            t1 = tc.tensor(1)
            t2 = tc.tensor(1) 
            clause = clauses[cla_id]
            for var_id in range(len(clause)):
                Vs = []
                Vu = []
                u_term = tc.tensor(1)
                s_term = tc.tensor(1)
                var_node = clause[var_id]
                nei_var_nodes = bpf.neiborhood_sat(var_node,clauses,max_item)
                tendency = tendencies[cla_id][var_id]
                for nei_var_node in nei_var_nodes:
                    clause_id = nei_var_node - max_item -1
                    var_node_id = clauses[clause_id].index(var_node)
                    if tendencies[clause_id][var_node_id]*tendency ==1 and nei_var_node != max_item+1+cla_id:
                        message_id = G_edges.index([var_node,nei_var_node])
                        s_term = s_term*(1-result_BP[message_id])
                    if tendencies[clause_id][var_node_id]*tendency ==-1 and nei_var_node != max_item+1+cla_id:
                        message_id = G_edges.index([var_node,nei_var_node])
                        u_term = u_term*(1-result_BP[message_id])
                t1 = t1*(u_term+s_term)
                t2 = t2*s_term
            Ws.append(t1-t2)
        return Ws

def EntropyBP(clauses,tendencies):
    G_fac,max_item = gg.G_generator_fac(clauses)
    G_edges,max_item,node_list,slice = bpf.G_generate_bpf(clauses)
    nodes = list(range(max_item+1))
    result_BP,_,step = bpf.belief_propagation_fast(G_edges,node_list,slice,clauses,tendencies,max_item)
    marginals_var = marginal_var_BP_entropy(G_edges,nodes,clauses,tendencies,result_BP,max_item)
    logs_var = []
    marginals_fac = marginal_fac_BP_entropy(G_edges,nodes,clauses,tendencies,result_BP,max_item)
    entropy4 = 0
    for var_node in range(len(marginals_var)):     
        entropy4 = entropy4 + (1-G_fac.degree(var_node))*tc.log(marginals_var[var_node])
        logs_var.append(tc.log(marginals_var[var_node]))
    for fac_node in range(len(marginals_fac)):
        entropy4 = entropy4 +tc.log(marginals_fac[fac_node])
        logs_var.append(tc.log(marginals_fac[fac_node]))
    return entropy4,step




