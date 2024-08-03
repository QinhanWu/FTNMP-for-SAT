import torch as tc
import opt_einsum as oe
import time
import numpy as np
import cotengra as ctg
import copy
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")



def local_contraction_sat_fac(G_fac,clauses,tendencies,max_item,device):
    """
    Perform contraction of a factor graph for a SAT problem.

    Parameters:
    G_fac (nx.Graph): The factor graph with nodes representing variables and clauses.
    clauses (list of lists): A list of clauses, where each clause is a list of variable indices.
    tendencies (list of lists): A list of tendencies, with each tendency being a list of three integers (-1 or 1).
    max_item (int): The maximum variable node index in the graph.
    device (torch.device): The device to which tensors are moved (e.g., CPU or CUDA).

    Returns:
    z (torch.Tensor): The result of contracting the factor graph tensors.
    norm (torch.Tensor): A normalization factor, representing the sum of logarithms of 2 for certain nodes.
    """
    tensors = []
    eqin = ''
    norm = tc.tensor(0.0)
    for noden in list(G_fac.nodes()):
            if noden>max_item:
                clause = clauses[noden - max_item-1]
                tensor = (tc.ones(size=[2]*len(clause))*(1.0)).to(device)
                zero_pos = [[int(-0.5*i+0.5)] for i in tendencies[noden - max_item-1]]
                tensor[zero_pos] = 0.0
                tensors.append(tensor.to(tc.float64))
                for fac_node in clause:
                    eqin += oe.get_symbol(fac_node)
                eqin += ","
            else:
                if noden>100:
                    tensor = tc.tensor([0.5,0.5]).to(device)
                    norm = norm+tc.log(tc.tensor(2.0))
                else:
                    tensor = tc.tensor([1.0,1.0]).to(device)
                tensors.append(tensor.to(tc.float64))
                eqin += oe.get_symbol(noden)
                eqin += ","
    eqin = eqin.rstrip(',')
    eqin += '->'
    z = oe.contract(eqin, *tensors, optimize=True)
    print(z.dtype)
    return z,norm


def local_contraction_ds(G_fac,clauses,tendencies,max_item,cc_limit,device):
    """
    Perform local contraction of a factor graph with specified constraints.

    Parameters:
    G_fac (nx.Graph): The factor graph with nodes representing variables and clauses.
    clauses (list of lists): A list of clauses, where each clause is a list of variable indices.
    tendencies (list of lists): A list of tendencies, with each tendency being a list of three integers (-1 or 1).
    max_item (int): The maximum variable node index in the graph.
    cc_limit (int): The limit for connected components in the contraction.
    device (torch.device): The device to which tensors are moved (e.g., CPU or CUDA).

    Returns:
    z (torch.Tensor): The result of contracting the factor graph tensors.
    sgn (int): The count of nodes with degree zero.
    T (object): The result of the custom contraction function.
    """
    tensors = []
    tensor_bonds = []
    eqin = ''
    norm = tc.tensor(0.0)
    for noden in sorted(list(G_fac.nodes())):
            if noden>max_item:
                clause = clauses[noden - max_item-1]
                tensor = (tc.ones(size=[2]*len(clause))*(1.0)).to(device)
                zero_pos = [[int(-0.5*i+0.5)] for i in tendencies[noden - max_item-1]]
                tensor[zero_pos] = 0.0
                tensors.append(tensor.to(tc.float64))
                for fac_node in clause:
                    eqin += oe.get_symbol(fac_node)
                eqin += ","
                tensor_bonds.append(clause)
    eqin = eqin.rstrip(',')
    eqin += '->'
    z,T = my_contract(eqin,tensors,tensor_bonds,cc_limit,max_item,device)
    sgn = 0
    for node in list(G_fac.nodes()):
        if G_fac.degree(node) == 0:
            sgn = sgn + 1
    return z,sgn,T


def my_contract(eq,tensors,tensor_bonds,cc_limit,max_item,device):
    """
    Contract tensors based on a given expression with optimization and dynamic slicing.

    Parameters:
    eq (str): The contraction expression.
    tensors (list of torch.Tensor): List of tensors to contract.
    tensor_bonds (list of lists): List of tensor bonds, indicating the connections between tensors.
    cc_limit (int): The limit for connected components.
    max_item (int): The maximum index for variable nodes.
    device (torch.device): The device to which tensors are moved (e.g., CPU or CUDA).

    Returns:
    z (torch.Tensor): The result of the contraction.
    T (float): The time taken for the contraction.
    """
    shapes = []
    symbol_list = []
    for i in list(range(max_item+1)):
        symbol1 = oe.get_symbol(i)
        symbol_list.append(symbol1)
    for tensor in tensors:
        shape = list(tensor.size())
        shapes.append(shape)
    print('contract exact')
    opt = ctg.HyperOptimizer(
        max_repeats=32,
        parallel=True,
        optlib="random",
        slicing_reconf_opts={
            "target_size": 2**25,},
        progbar=True,
        )
    path, info = oe.contract_path(eq, *shapes, shapes=True, optimize=opt)
    tree_s = opt.best['tree']
    sliced_inds = list(tree_s.sliced_inds)
    #print('sliced_inds',sliced_inds)    
    for i in sliced_inds:
        eq = eq.replace(i, '')
    slicing_edges = []
    for ind in list(tree_s.sliced_inds):
        bond_id = symbol_list.index(ind)
        slicing_edges.append(bond_id)
    #print('slicing_edges:',slicing_edges)
    t1 = time.time()
    slicing_indices = {}.fromkeys(slicing_edges)
    bonds_copy = copy.deepcopy(tensor_bonds)
    for idx in slicing_edges:
        slicing_indices[idx] = []
        for x in range(len(bonds_copy)):
            if idx in bonds_copy[x]:
                slicing_idx = bonds_copy[x].index(idx)  
                slicing_indices[idx].append((x, slicing_idx))
                bonds_copy[x].pop(slicing_idx)
    configs = []
    for s in tqdm(range(2**(len(sliced_inds))),disable=True):
       
        s1 = list(map(int, np.binary_repr(s,len(sliced_inds))))  
        config = copy.deepcopy(s1)
        configs.append(config)
    z = tc.tensor(0,dtype=tc.double).to(device)
    for config_id in tqdm(range(len(configs))):
        config = configs[config_id]
        tensors_n = copy.deepcopy(tensors)
        for x in range(len(slicing_edges)):
            idx = slicing_edges[x]
            for id, slicing_idx in slicing_indices[idx]:
                tensors_n[id] = (tensors_n[id].select(slicing_idx, config[x]).clone()).to(device)   
        eq_n = copy.deepcopy(eq)
        z += oe.contract(eq_n, *tensors_n, optimize=path)
        t2 = time.time()
        T = t2-t1
    return z,T









