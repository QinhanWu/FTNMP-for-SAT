import torch as tc
import opt_einsum as oe




def local_contraction_sat_fac(G_fac,clauses,tendencies,max_item,device):
    tensors = []
    eqin = ''
    norm = tc.tensor(0)
    for noden in list(G_fac.nodes()):
        if noden>max_item:
            clause = clauses[noden - max_item-1]
            #print(noden,clause)
            tensor = (tc.ones(size=[2]*len(clause))*(1.0)).to(device)
            zero_pos = [[int(-0.5*i+0.5)] for i in tendencies[noden - max_item-1]]
            tensor[zero_pos] = 0.0
            tensors.append(tensor)
            #因素节点对于的腿安排好
            for fac_node in clause:
                eqin += oe.get_symbol(fac_node)
            eqin += ","
        else:#如果变量节点在边缘上，无论有多离谱的外界情况，传进来的肯定是一个向量
            if noden>100:
                tensor = tc.tensor([0.5,0.5]).to(device)
                norm = norm+tc.log(tc.tensor(2))
            else:
                tensor = tc.tensor([1.0,1.0]).to(device)
            tensors.append(tensor)
            eqin += oe.get_symbol(noden)
            eqin += ","
    eqin = eqin.rstrip(',')
    eqin += '->'
    #print(tensors)
    #print(eq)
    z = oe.contract(eqin, *tensors, optimize=True)
    #print(z)
    return z,norm

