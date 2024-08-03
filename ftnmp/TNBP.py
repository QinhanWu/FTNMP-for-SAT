import torch as tc
import networkx as nx
import copy 
import opt_einsum as oe
import time

def G_generator_fac(clauses):
    """
    Generate a graph based on the given list of clauses and compute the maximum node identifier.

    Parameters:
    clauses (list of lists): Each clause is a list of integers representing nodes connected by edges in the graph. Nodes within a clause are connected to an additional node representing the clause.

    Returns:
    Returns:
    G (nx.Graph): The generated undirected graph where nodes and edges are defined by the clauses.
    max_item (int): The highest node identifier in the graph, which is the maximum variable node index across all clauses plus the number of clauses.

    """
    G = nx.Graph()    
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

def local_contraction_sat_fac(G_fac,Ne_cavity,clauses,tendencies,cavity,node,max_item,egdelist2,device):
    """
    Perform local tensor contraction for a specific node in a graph, esing tensors and equations based on the node's cavity sungraph and clauses.

    Parameters:
    G_fac (nx.Graph): The factor graph representing the problem.
    Ne_cavity (list of int): List of nodes in the cavity.
    clauses (list of lists): List of clauses, where each clause is a list of integers representing nodes.
    tendencies (list of lists): List of tendencies for nodes, indicating which tensor positions should be set to zero.
    cavity (list of dicts): List of dictionaries containing tensors for the cavity nodes.
    node (int): The current node to perform contraction on.
    max_item (int): The maximum node identifier used in the graph.
    egdelist2 (list of lists): List of edges in the graph, where each edge is represented as a list of two nodes.
    device (torch.device): The device (CPU or GPU) on which tensors are allocated.

    Returns:
    z (torch.Tensor): The result of the tensor contraction, adjusted for cases where the resulting tensor is zero.
    """
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
    """
    Perform local tensor contraction for a specific node in a graph using a provided contraction function.

    Parameters:
    Ne_cavity (list of int): List of nodes in the cavity.
    clauses (list of lists): List of clauses, where each clause is a list of integers representing nodes.
    tendencies (list of lists): List of tendencies for nodes, indicating which tensor positions should be set to zero.
    cavity (list of dicts): List of dictionaries containing tensors for the cavity nodes.
    node (int): The current node to perform contraction on.
    max_item (int): The maximum node identifier used in the graph.
    egdelist2 (list of lists): List of edges in the graph, where each edge is represented as a list of two nodes (not used in this function).
    my_contract (function): Function to perform the tensor contraction.
    device (torch.device): The device (CPU or GPU) on which tensors are allocated.

    Returns:
    z (torch.Tensor): The result of the tensor contraction, adjusted for cases where the resulting tensor is zero.
    """
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
    """
    Perform an initial local tensor contraction for a specific node in a graph and prepare a contraction path for future use.

    Parameters:
    G_fac (nx.Graph): The factor graph representing the problem.
    Ne_cavity (list of int): List of nodes in the cavity.
    clauses (list of lists): List of clauses, where each clause is a list of integers representing nodes.
    tendencies (list of lists): List of tendencies for nodes, indicating which tensor positions should be set to zero.
    cavity (list of dicts): List of dictionaries containing tensors for the cavity nodes.
    node (int): The current node to perform contraction on.
    max_item (int): The maximum node identifier used in the graph.
    egdelist2 (list of lists): List of edges in the graph, where each edge is represented as a list of two nodes.
    device (torch.device): The device (CPU or GPU) on which tensors are allocated.

    Returns:
    z (torch.Tensor): The result of the tensor contraction, adjusted for cases where the resulting tensor is zero.
    my_con (function): A function for performing the contraction path.
    """
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
                for nodet in nodes_subset:
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
    if len(z.size())== 1:
       z = (tc.tensor([z[0],z[1],0,0])).to(device)
    if len(z.size())== 2:
       z=(z.reshape(4)).to(device) 
    con = (tc.zeros(4)).to(device)
    if tc.equal(z,con)==True:
        z =(tc.tensor([0.5,0.5,0.5,0.5])).to(device)

    
    return z,my_con

def inner_node(node, region_id,region_bound):
    """
    Determine if a node is an inner node within a specific region.

    Parameters:
    node (int): The node identifier to check.
    region_id (int): The identifier of the region in which to check the node.
    region_bound (list of lists): List of boundaries, where each region is a list of integers representing nodes in its boundary.

    Returns:
    bool: `True` if the node is not in the boundary of the specified region, indicating it is an inner node. `False` otherwise.
    """
    if node in region_bound[region_id]:
        return False
    else:
        return True

def TNBP_3_sat_interaction_new(G_fac,Nv,boundaries,clauses,tendencies,interactions,max_item,region_info,region_bound,device):
    """
    Perform a tensor network belief propagation (the message here is equal to belief --the probapalities of nodes satisfy the constraints) 
    algorithm for a 3-SAT problem with interaction.

    Parameters:
    G_fac (nx.Graph): The factor graph representing the problem.
    Nv (list of lists): List of nodes in the neighborhoods of the corresponding node of index.
    boundaries (list of lists): List of nodes in the boundary of the neighborhoods of the corresponding node of index.
    clauses (list of lists): List of clauses, where each clause is a list of integers representing nodes.
    tendencies (list of lists): List of tendencies for nodes, indicating which tensor positions should be set to zero.
    interactions (list of lists): List of interactions, where each interaction is a list of nodes.
    max_item (int): The maximum node identifier used in the graph.
    region_info (list of lists): List of regions, where each region is a list of node identifiers.
    region_bound (list of lists): List of boundaries, where each region is a list of integers representing nodes in its boundary.
    device (torch.device): The device (CPU or GPU) on which tensors are allocated.

    Returns:
    cavity (torch.Tensor): The updated cavity tensor after the belief propagation.
    converged (bool): `True` if the algorithm converged within the step limit, `False` otherwise.
    egdelist2 (list of lists): The edge list of the factor graph, sorted.
    step (int): The number of steps performed before convergence or reaching the step limit.
    """
    converged = True
    n = G_fac.number_of_nodes()
    step_limit = 1000
    epsilon = 1e-3
    difference_max = tc.tensor([0.0]).to(device)
    damping_factor = 0         #The damping is adjustable.
    egdelist2  = [sorted(edge) for edge in G_fac.edges()]
    cavity = (tc.ones(size=(n,n,4))/2).to(device)
    paths = [[[]for w in range(n)] for d in range(n)]
    t1 = time.time()
    for step in range(step_limit):
        first_center_node = -1

        for div_id in range(len(region_info)):
            div = region_info[div_id]
            inner_msg = {}
            for center_node in div:
                first_boundary_node = -1
                neighborhood = Nv[center_node]
                if (len(boundaries[center_node]) != 0 and first_center_node == -1):
                    first_center_node = center_node

                for node in neighborhood:
                    is_inner_msg = inner_node(center_node, div_id,region_bound) and not inner_node(node, div_id,region_bound)
                    if (node  in inner_msg) and is_inner_msg:
                        cavity[node][center_node] = inner_msg[node]
                        continue

                    if node in boundaries[center_node]:
                        if first_boundary_node == -1:
                            first_boundary_node = node
                        Ne_cavity = list(set(Nv[node]).difference(Nv[center_node]))

                        for interaction in interactions:
                            for adn in interaction:
                                if node == adn:
                                    for adc in interaction:
                                        if adc in Nv[center_node] and adc != adn and adc not in Ne_cavity and adc != center_node  :
                                                Ne_cavity.append(adc)
                                                Ne_cavity = list(set(Ne_cavity))
                        if len(Ne_cavity)!=0:
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
                            if is_inner_msg:
                                inner_msg[node] = temp
                        else:
                            difference =tc.tensor(0.0).to(device) 
                    else:
                        cavity[node][center_node] = tc.tensor([1.0,1.0,0,0]).to(device)
                        cavity[node][center_node] = cavity[node][center_node]/tc.sum(cavity[node][center_node])
                        difference =tc.tensor(0.0).to(device)

                    if center_node == first_center_node and node == first_boundary_node:
                        difference_max = difference
                    elif difference > difference_max:
                        difference_max = difference
        t2 = time.time()
        print("iteration step:",step+1,",   difference:",difference_max.item(),'    time:',t2-t1 )
        if difference_max <= epsilon :
            break   
    if step==step_limit-1:
        print('unconverged') 
        converged = False  
    return cavity ,converged,egdelist2, step+1