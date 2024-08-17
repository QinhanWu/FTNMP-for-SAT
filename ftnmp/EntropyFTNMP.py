import torch as tc
import opt_einsum as oe
import TNBP as TNBP
import nx_test as nxt
import get_regions_new as grn


def entropy_single_point(single_point,G_fac,clauses,tendencies,devides,devide_bound,cavity,egdelist2,degree_node,max_item,device):
    '''
    -----------------------------------
    It can be proven that when 𝑛 regions are connected by only one node, the impact of this node on entropy is 
    (𝑛-1)×ln(product of the outgoing messages of this node)
    -----------------------------------
    
    Computes the entropy contribution of a single point node in a factor graph, given its impact on the entropy of regions connected through this node.

    Parameters:
    - single_point (int): The node whose entropy impact is being calculated.
    - G_fac (Graph): The factor graph where nodes are connected by factors.
    - clauses (list of lists): List of clauses, where each clause is a list of variable nodes.
    - tendencies (list of lists): List of tendencies corresponding to each clause.
    - devides (list of lists): List of node sets, where each set corresponds to a division of the factor graph.
    - devide_bound (list of lists): List of boundaries corresponding to each division.
    - cavity (tensor): A tensor with shape (n+m)*(n+m), where n is the number of variable node and m is the number of factor node, to store the cavity messages form the first axis to the second axis.
    - egdelist2 (list of lists): List of edges, where each edge connects two nodes.
    - degree_node (dict): Dictionary where keys are nodes and values are their degrees.
    - max_item (int): The maximum index of nodes in the factor graph.
    - device (torch.device): The device on which to perform tensor operations (e.g., 'cpu' or 'cuda').

    Returns:
    - Error (Tensor): The computed entropy error for the single point node.
    
    '''
    tensors = []
    #norms = []
    eq = ''
    for bd_id in range(len(devide_bound)):
        bd = devide_bound[bd_id]
        if single_point in bd:
            devide = devides[bd_id]
            subtensors = []
            subeqin = ''
            for node in devide:
                if node>max_item and node - max_item-1<len(clauses):
                    clause = clauses[node - max_item-1]
                    subtensor = (tc.ones(size=[2]*len(clause))*(1.0)).to(device)
                    zero_pos = [[int(-0.5*i+0.5)] for i in tendencies[node - max_item-1]]
                    subtensor[zero_pos] = 0.0
                    subtensors.append(subtensor)
                    for node_in_clause in clause:
                        Q = set(G_fac.neighbors(node_in_clause))
                        R = set(devide)
                        if len(Q.intersection(R))>1 and node in bd and node_in_clause not in devide:
                            symbol_id = egdelist2.index([node_in_clause,node])
                            subeqin +=oe.get_symbol(symbol_id+max_item+1)
                        else:
                            subeqin += oe.get_symbol(node_in_clause)
                    subeqin += ","
                if  G_fac.degree(node)==1 :
            
                    subtensor = (tc.tensor([0.5,0.5])).to(device)
                    subeqin += oe.get_symbol(node)
                    subtensors.append(subtensor)
                    subeqin += "," 

            for nodey in [i for i in bd if i != single_point]:
                nodey_nei = list(G_fac.neighbors(nodey))
                message = sorted([item for item in nodey_nei if item not in devide],reverse=False)
                Nv_pre = [(n,len(devide)) for n in list(set(nodey_nei).difference(set(message)))]    
                max_tuple = max(Nv_pre, key=lambda x: x[1])
                cavity_id = devide.index(max_tuple[0])
                tensor_pre = cavity[nodey][devide[cavity_id]]
                if len(message)==1:
                    subtensor =(tc.tensor([tensor_pre[0],tensor_pre[1]])).to(device)
                    subtensors.append(subtensor)
                    if message[0]<=max_item:
                        Q = set(G_fac.neighbors(message[0]))
                        R = set(devide)
                        if len(Q.intersection(R))>1:
                            symbol_id = egdelist2.index([message[0],nodey])
                            subeqin +=oe.get_symbol(symbol_id+max_item+1)
                        else:
                            subeqin += oe.get_symbol((message[0]))
                    else:
                        subeqin += oe.get_symbol(nodey)
                        subeqin += ","
                else:
                    if nodey>max_item:
                        subtensor = (tensor_pre.reshape(2,2)).to(device)
                        subtensors.append(subtensor)
                        for message_id in range(len(message)):
                            Q = set(G_fac.neighbors(message[message_id]))
                            R = set(devide)
                            if len(Q.intersection(R))>1:
                                symbol_id = egdelist2.index([message[message_id],nodey])
                                subeqin +=oe.get_symbol(symbol_id+max_item+1)
                            else:
                                subeqin += oe.get_symbol((message[message_id]))
                        subeqin += ","
                    else:
                        subtensor = (tc.tensor([tensor_pre[0],tensor_pre[1]])).to(device)
                        subtensors.append(subtensor)
                        subeqin += oe.get_symbol(nodey)
                        subeqin += ","
            subeqin = subeqin.rstrip(',') 
            subeqin += '->'
            
            subeqin += oe.get_symbol(single_point)
            subz = oe.contract(subeqin,*subtensors,optimize=True)
            subz /= tc.sum(subz)
            tensors.append(subz)
            eq += oe.get_symbol(single_point)
            eq += ','

    eq = eq.rstrip(',') 
    eq +='->'
    z = oe.contract(eq,*tensors)
    z = ((degree_node[single_point]-1))*tc.log(z)
    z_errors =[]
    for i in range(len(tensors)):
        tensors_error = [tensors[j] for j in range(len(tensors)) if j != i]
        eq_error = ''
        for k in range(len(tensors_error)):
            eq_error += 'a,'
        eq_error = eq_error.rstrip(',')
        eq_error += '->'
        if len(tensors_error) != 0 :
            z_error  = oe.contract(eq_error,*tensors_error,optimize =True)
            z_errors.append(z_error)
    Error = z - tc.log(tc.prod(tc.tensor(z_errors)))
    return Error

def entropy_cluster(tar_node,G_fac,Nv,boundaries,clauses,tendencies,cavity,max_item,nei_cxx,device):
    '''
    Computes the local entropy of a cluster of nodes in a factor graph related to a target node.

    Parameters:
    - tar_node (int): The target node for which entropy is being calculated.
    - G_fac (Graph): The factor graph where nodes are connected by factors.
    - Nv (list of lists): List of devides of the corresponding of index.
    - boundaries (list of lists): List of nodes in the boundary of devides of the corresponding node of index.
    - clauses (list of lists): List of clauses, where each clause is a list of variable nodes.
    - tendencies (list of lists): Tendencies for each clause.
    - cavity (tensor): A tensor with shape (n+m)*(n+m), where n is the number of variable node and m is the number of factor node, to store the cavity messages form the first axis to the second axis.
    - max_item (int): Maximum index value in the graph.
    - nei_cxx (list of lists): List of nodes in the neighborhoods of the corresponding node of index.
    - device (torch.device): The device on which to perform tensor operations (e.g., 'cpu' or 'cuda').

    Returns:
    - marginal (Tensor): The computed !entropy! tensor for the cluster related to the target node.
    '''
    egdelist2 = [sorted(edge) for edge in G_fac.edges()]
    tensors = []
    eqin = ''
    for noden in Nv[tar_node]:
        if noden > max_item and noden - max_item-1<len(clauses):
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
        if len(Nv[tar_node])!=2 and G_fac.degree(noden)==1 and noden not in boundaries[tar_node]:
            tensor = (tc.tensor([0.5,0.5])).to(device)
            eqin += oe.get_symbol(noden)
            tensors.append(tensor)
            eqin += "," 
        
    for nodey in boundaries[tar_node]:
        nodey_nei = list(G_fac.neighbors(nodey))
        message = sorted([item for item in nodey_nei if item not in Nv[tar_node]],reverse=False)
        Nv_pre = [(n,len(nei_cxx[n])) for n in list(set(nodey_nei).difference(set(message)))]
        max_tuple = max(Nv_pre, key=lambda x: x[1])
        cavity_id = Nv[tar_node].index(max_tuple[0])
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
    eq = eqin 
    if len(tensors)!=0:
        marginal = oe.contract(eq, *tensors, optimize=True)
    else:
        marginal = (tc.tensor(2)).to(device)
    return marginal

def entropies_cluster(G_fac,clauses,tendencies,devide,devide_bound,max_item,nei_cxx,cavity,device):
    '''
    Computes the entropy for each cluster in the factor graph and returns the logarithms of the marginal probabilities.

    Parameters:
    - G_fac (Graph): The factor graph where nodes are connected by factors.
    - clauses (list of lists): List of clauses, where each clause is a list of variable nodes.
    - tendencies (list of lists): Tendencies for each clause.
    - devide (list of lists): List of clusters, where each cluster is a list of nodes.
    - devide_bound (list of lists): List of boundaries for each cluster.
    - max_item (int): Maximum index value in the graph.
    - nei_cxx (list of lists): List of nodes in the neighborhoods of the corresponding node of index.
    - cavity (tensor): A tensor with shape (n+m)*(n+m), where n is the number of variable node and m is the number of factor node, to store the cavity messages form the first axis to the second axis.
    - device (torch.device): The device on which to perform tensor operations (e.g., 'cpu' or 'cuda').

    Returns:
    - marginal_devides (list of Tensor): A list of tensors where each tensor is the logarithm of the local entropy for a cluster.
    '''
    marginal_devides = []
    for devide_id in range(len(devide)):
        marginal_devide = entropy_cluster(devide_id,G_fac,devide,devide_bound,clauses,tendencies,cavity,max_item,nei_cxx,device)
        marginal_devides.append(tc.log(marginal_devide))
    return marginal_devides

def entropies_point(single_list,G_fac,clauses,tendencies,devides,devide_bound,cavity,edgelist2,degree_node,max_item,device):
    '''
    Computes the entropy for a list of nodes in the factor graph and returns the influence to the global entropies.

    Parameters:
    - single_list (list of int): List of nodes for which to compute entropy.
    - G_fac (Graph): The factor graph where nodes are connected by factors.
    - clauses (list of lists): List of clauses, where each clause is a list of variable nodes.
    - tendencies (list of lists): Tendencies for each clause.
    - devides (list of lists): List of clusters, where each cluster is a list of nodes.
    - devide_bound (list of lists): List of boundaries for each cluster.
    - cavity (tensor): A tensor with shape (n+m)*(n+m), where n is the number of variable node and m is the number of factor node, to store the cavity messages form the first axis to the second axis.
    - edgelist2 (list of lists): List of edges in the factor graph.
    - degree_node (dict): Dictionary where keys are nodes and values are their degrees.
    - max_item (int): Maximum index value in the graph.
    - device (torch.device): The device on which to perform tensor operations (e.g., 'cpu' or 'cuda').

    Returns:
    - logs_point (list of Tensor): A list of tensors where each tensor is the entropy for a node in the `single_list`.
    '''
    logs_point = []
    for node in set(single_list):
        if node in degree_node:
            m = entropy_single_point(node,G_fac,clauses,tendencies,devides,devide_bound,cavity,edgelist2,degree_node,max_item,device)
            logs_point.append(m)
    return logs_point

def entropy_intersection(G_fac,edges,clauses,tendencies,cavity,max_item,Nv,device):
    '''
    Computes the entropy for intersections of clusters defined by the edges in the factor graph.

    Parameters:
    - G_fac (Graph): The factor graph where nodes are connected by factors.
    - edges (list of lists): List of edges in the factor graph.
    - clauses (list of lists): List of clauses, where each clause is a list of variable nodes.
    - tendencies (list of lists): Tendencies for each clause.
    - cavity (tensor): A tensor with shape (n+m)*(n+m), where n is the number of variable node and m is the number of factor node, to store the cavity messages form the first axis to the second axis.
    - max_item (int): Maximum index value in the graph.
    - Nv (list of lists): List of devides of the corresponding of index.
    - device (torch.device): The device on which to perform tensor operations (e.g., 'cpu' or 'cuda').

    Returns:
    - ms (list of Tensor): A list of tensors where each tensor is the logarithm of entropy for each edge intersection.
    '''
    ms = []
    for edge_id in range(len(edges)):
        m = entropy_cluster(edge_id,G_fac,edges,edges,clauses,tendencies,cavity,max_item,Nv,device)
        ms.append(tc.log(m))
    return ms

def EntropyZ_new(G_fac,clauses,tendencies,devides,devide_bound,region_info,single_list,max_item,Nv,boundaries,degree_node,device,file):
    '''
    Computes the entropy of a system using various clusters and intersections defined in the factor graph.

    Parameters:
    - G_fac (Graph): The factor graph where nodes are connected by factors.
    - clauses (list of lists): List of clauses, where each clause is a list of variable nodes.
    - tendencies (list of lists): Tendencies for each clause.
    - devides (list of lists): Decomposed sets of nodes for entropy calculations.
    - devide_bound (list of lists): Boundaries for each decomposition.
    - region_info (list): Information about different regions in the graph.
    - single_list (list): List of single nodes to consider for point-wise entropy calculations.
    - max_item (int): Maximum index value in the graph.
    - Nv (dict): Dictionary where keys are nodes and values are sets of neighboring nodes.
    - boundaries (dict): Dictionary where keys are nodes and values are boundary sets.
    - degree_node (dict): Dictionary where keys are nodes and values are their degrees.
    - device (torch.device): The device on which to perform tensor operations (e.g., 'cpu' or 'cuda').
    - file (str): Path to the file where results may be saved or used.

    Returns:
    - entropy (Tensor): The computed entropy of the system.
    - step (int): The number of steps taken in the convergence process.
    '''

    intersections_info, edges_info, direct_products_info ,else_situations =nxt.cluster_edges_generate_new(G_fac,max_item,devides,devide_bound)
    direct_products = [sorted(direct_products_info[i][2]) for i in range(len(direct_products_info)) if len(direct_products_info[i][2])>=2 and min(direct_products_info[i][0],direct_products_info[i][1])<=len(region_info)]
    region_info,region_bound = grn.region_info_change(G_fac,clauses,region_info,max_item,single_list)
    cavity,converged,edgelist2,step = TNBP.TNBP_3_sat_interaction_new(G_fac,Nv,boundaries,clauses,tendencies,direct_products,max_item,region_info,region_bound,device)
    intersections = [sorted(edges_info[i][2]) for i in range(len(edges_info))  ] 
    loges_devides =  entropies_cluster(G_fac,clauses,tendencies,devides,devide_bound,max_item,Nv,cavity,device)
    logs_points = entropies_point(single_list,G_fac,clauses,tendencies,devides,devide_bound,cavity,edgelist2,degree_node,max_item,device)
    ms = sum(entropy_intersection(G_fac,intersections,clauses,tendencies,cavity,max_item,Nv,device))
    entropy = sum(loges_devides)-sum(logs_points)-ms
   

    return entropy,step
