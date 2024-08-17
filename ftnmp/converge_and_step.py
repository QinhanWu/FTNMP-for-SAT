import torch as tc
import graph_generate as gg
import numpy as np
import time
import EntropyBP as EBP
import EntropyFTNMP as EFT
import get_regions_new as grn
import TNBP
import nx_test as nxt
import BP_fast as bpf
import copy

def TNBP_3_sat_interaction_new(G_fac,Nv,boundaries,clauses,tendencies,interactions,max_item,region_info,region_bound,damping_factor,device):
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
    damping_factor(float): A parameter that adjusts the rate of iterative updates
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
                    is_inner_msg = TNBP.inner_node(center_node, div_id,region_bound) and not TNBP.inner_node(node, div_id,region_bound)
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
                                        if adc in Nv[center_node] and adc != adn and adc not in Ne_cavity and adc != center_node :
                                                Ne_cavity.append(adc)
                                                Ne_cavity = list(set(Ne_cavity))
                        if len(Ne_cavity)!=0:
                            if step > 0 :
                                New_cavity_vec = TNBP.local_contraction(Ne_cavity,clauses,tendencies,cavity,node,max_item,egdelist2,paths[node][center_node],device)
                            else:
                                New_cavity_vec,my_contract = TNBP.local_contraction_begin(G_fac,Ne_cavity,clauses,tendencies,cavity,node,max_item,egdelist2,device)
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
        converged = False  
    return cavity ,converged,egdelist2, step+1

def belief_propagation_fast(G_edges,node_list,slice,clauses,tendencies,max_item,damping):
    """
    Perform belief propagation on a factor graph to compute messages between factor and variable nodes.

    Parameters:
        G_edges (list of lists of int): List of edges in the graph, where each edge is represented as a list of two integers.
        node_list (list of int): List of unique node indices in the graph.
        slice (list of lists of int): List of slices for each node.
        clauses (list of lists of int): List of clauses, where each clause is represented as a list of integer literals.
        tendencies (list of lists of int): List of tendencies for each clause, where each tendency is represented as a list of integer values.
        max_item (int): Maximum index for qubit nodes, used to differentiate between qubit nodes and factor nodes.
        damping_factor(float): A parameter that adjusts the rate of iterative updates.

    Returns:
        message_fac_to_var (tensor): Final messages from factor nodes to variable nodes.
        success_or_not (str): Status indicating whether the algorithm converged or not.
        num_iterations (int): Number of iterations performed before convergence or reaching the iteration limit.
    """
    interation = 1000
    epsilon = 1e-3
    message_fac_to_var = tc.ones(len(G_edges))
    message_fac_to_var=message_fac_to_var/2
    information = bpf.information_of_fac(G_edges,clauses,tendencies,max_item)
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


def EntropyBP(clauses,tendencies,damping):
    """
    Compute the entropy of a factor graph using belief propagation.

    Parameters:
    clauses (list of lists): List of clauses, where each clause is a list of variable nodes.
    tendencies (list of lists): List of tendencies for each clause, where each tendency list corresponds to a clause.
    damping_factor(float): A parameter that adjusts the rate of iterative updates.

    Returns:
    entropy (tensor): Total entropy computed for the factor graph.
    step (int): Number of iterations performed during belief propagation.
    """
    G_fac,max_item = gg.G_generator_fac(clauses)
    G_edges,max_item,node_list,slice = bpf.G_generate_bpf(clauses)
    nodes = list(range(max_item+1))
    result_BP,_,step = belief_propagation_fast(G_edges,node_list,slice,clauses,tendencies,max_item,damping)
    marginals_var = EBP.marginal_var_BP_entropy(G_edges,nodes,clauses,tendencies,result_BP,max_item)
    logs_var = []
    marginals_fac = EBP.marginal_fac_BP_entropy(G_edges,nodes,clauses,tendencies,result_BP,max_item)
    entropy4 = 0
    for var_node in range(len(marginals_var)):     
        entropy4 = entropy4 + (1-G_fac.degree(var_node))*tc.log(marginals_var[var_node])
        logs_var.append(tc.log(marginals_var[var_node]))
    for fac_node in range(len(marginals_fac)):
        entropy4 = entropy4 +tc.log(marginals_fac[fac_node])
        logs_var.append(tc.log(marginals_fac[fac_node]))
    return entropy4,step

def converge_and_step(G_fac,clauses,tendencies,devides,devide_bound,region_info,single_list,max_item,Nv,boundaries,degree_node, device,damping_factor = None):
    
    """
    This program tests the number of steps. FTNMP is typically not set to R = 2 because its accuracy is similar to BP. In practice, 
    since various types of graphs might be encountered often have cycles affecting convergence, a damping factor is needed.
    We will use a recommended damping factor of 0.5 for testing if R = 2 and damping factor of 0 for else cases.
    -----------------------------------------------------------------------------------------------------------------------------
    Parameters:
    - G_fac (Graph): The factor graph where nodes are connected by factors.
    - clauses (list of lists): List of clauses, where each clause is a list of variable nodes.
    - tendencies (list of lists): Tendencies for each clause.
    - devides (list of lists): Decomposed sets of nodes for entropy calculations.
    - devide_bound (list of lists): Boundaries for each decomposition.
    - region_info (list): Information about different regions in the graph.
    - single_list (list): List of single nodes to consider for point-wise entropy calculations.
    - max_item (int): Maximum index value in the graph.
    - Nv (list of lists): List of devides of the corresponding of index.
    - boundaries (list of lists): List of nodes in the boundary of the neighborhoods of the corresponding node of index.
    - degree_node (dict): Dictionary where keys are nodes and values are their degrees.
    - device (torch.device): The device on which to perform tensor operations (e.g., 'cpu' or 'cuda').
    - damping_factor(float): Used to control the update learning rate. If not explicitly specified, the damping factor is set to 0.5 for R=2 and R=3, and 0 for all other cases.

    Returns:
    - entropy (Tensor): The computed entropy of the system.
    - step (int): The number of steps taken in the convergence process.

    """
    if damping_factor is None:
        if R_region == 2 or R_region == 3:
            damping_factor = 0.5
        else:
            damping_factor = 0

    intersections_info, edges_info, direct_products_info ,else_situations =nxt.cluster_edges_generate_new(G_fac,max_item,devides,devide_bound)
    direct_products = [sorted(direct_products_info[i][2]) for i in range(len(direct_products_info)) if len(direct_products_info[i][2])>=2 and min(direct_products_info[i][0],direct_products_info[i][1])<=len(region_info)]
    region_info,region_bound = grn.region_info_change(G_fac,clauses,region_info,max_item,single_list)
    cavity,converged,edgelist2,step = TNBP_3_sat_interaction_new(G_fac,Nv,boundaries,clauses,tendencies,direct_products,max_item,region_info,region_bound,damping_factor,device)
    intersections = [sorted(edges_info[i][2]) for i in range(len(edges_info))  ] 
    loges_devides =  EFT.entropies_cluster(G_fac,clauses,tendencies,devides,devide_bound,max_item,Nv,cavity,device)
    logs_points = EFT.entropies_point(single_list,G_fac,clauses,tendencies,devides,devide_bound,cavity,edgelist2,degree_node,max_item,device)
    ms = sum(EFT.entropy_intersection(G_fac,intersections,clauses,tendencies,cavity,max_item,Nv,device))
    entropy = sum(loges_devides)-sum(logs_points)-ms

    return step


device = 'cuda:0'
R_subregion = 100
file = open('step.txt','w')

times =[]
steps =[]

seed =0
oom = 0
for i in range(2):
    t = []
    step=[]
    nodes = 300
    alpha = 0.7
    n_cl = 0
    alpha_local=[2.5,3.2]
    max_cl = 12
    min_cl = 8
    clauses,tendencies = gg.double_random_generate(nodes,alpha,n_cl,alpha_local,min_cl,max_cl,seed)
    print('random seed:',seed,file = file)
    print('nodes:',nodes,'alpha_grobal:',alpha,'number of cluster:',n_cl,'alpha_local:',alpha_local,'Maximum points:',max_cl,'Minimum points:',min_cl,file = file)
    print('clauses=',clauses,file =file)
    print('tendencies=',tendencies,file = file)
    print('alpha:',len(clauses)/nodes,file = file)
    G_fac,max_item =gg.G_generator_fac(clauses)
    nodes = list(range(max_item+1))
    try:
        print('\n','BP_damping = 0.5:')
        t5 = time.time()
        _,step_BP_05 = EntropyBP(clauses,tendencies,0.5)
        t6 = time.time()
        t.append(t6-t5)
        step.append(step_BP_05)

        print('\n','BP_damping = 0:')
        t9 = time.time()
        _,step_BP_0 = EntropyBP(clauses,tendencies,0)
        t10 = time.time()
        t.append(t10-t9)
        step.append(step_BP_0)
        
        R_region = 2
        devides,subdevides,region_info,single_list = grn.get_regions_z(G_fac,clauses,max_item,R_region,R_subregion)
        Nv,boundaries = grn.devides_to_Nv(G_fac,devides,region_info)
        devide_bound,degree_node = grn.devides_others(G_fac,devides)
        print('\n','FTNBP_2:')
        t11 = time.time()
        step_2 = converge_and_step(G_fac,clauses,tendencies,devides,devide_bound,region_info,single_list,max_item,Nv,boundaries,degree_node, device,damping_factor = None)
        t12 = time.time()
        t.append(t12-t11)
        step.append(step_2)
    
        R_region = 4
        devides,subdevides,region_info,single_list = grn.get_regions_z(G_fac,clauses,max_item,R_region,R_subregion)
        Nv,boundaries = grn.devides_to_Nv(G_fac,devides,region_info)
        devide_bound,degree_node = grn.devides_others(G_fac,devides)
        print('\n','FTNBP_4:')
        t7 = time.time()
        step_4 =converge_and_step(G_fac,clauses,tendencies,devides,devide_bound,region_info,single_list,max_item,Nv,boundaries,degree_node, device,damping_factor = None)
        t8 = time.time()
        t.append(t8-t7)
        step.append(step_4)
        
        R_region = 6
        devides,subdevides,region_info,single_list = grn.get_regions_z(G_fac,clauses,max_item,R_region,R_subregion)
        Nv,boundaries = grn.devides_to_Nv(G_fac,devides,region_info) 
        devide_bound,degree_node = grn.devides_others(G_fac,devides)
        print('\n','FTNBP_6:')
        t3 = time.time()
        step_6 = converge_and_step(G_fac,clauses,tendencies,devides,devide_bound,region_info,single_list,max_item,Nv,boundaries,degree_node, device,damping_factor = None)
        t4 = time.time()
        t.append(t4-t3)
        step.append(step_6)

        times.append(t)
        steps.append(step)
    
      
        print('_________________________________average time_______________________________________',file=file)
        print(times,file = file)
        T = np.array(times)
        column_means = np.mean(T, axis=0)
        print('EXACT---BP_05---BP_0---E2---E4---E6----Tbar:', '\n',column_means,file=file)
        max_valuest = np.max(T, axis=0)
        min_valuest = np.min(T, axis=0)
        resultt = list(zip(min_valuest.reshape(1, -1)[0], max_valuest.reshape(1, -1)[0]))
        print('EXACT---BP_05---BP_0---E2---E4---E6----T(min,max):', '\n',resultt,file=file)
        print('_______________________________________steps_______________________________________',file=file)
        print(steps,file = file)
        STEP  = np.array(steps)
        column_means_steps = np.mean(STEP,axis=0)
        print('BP_05---BP_0---E2---E4---E6----Stepsbar:', '\n',column_means_steps,file=file)
        max_valuesp = np.max(STEP, axis=0)
        min_valuesp = np.min(STEP, axis=0)
        resultp= list(zip(min_valuesp.reshape(1, -1)[0], max_valuesp.reshape(1, -1)[0]))
        print('BP_05---BP_0---E2----E4---E6----Steps(min,max):', '\n',resultp,file=file)
        
        #print('time:',T)
        print('\n','\n',file = file)

        #print(f"Step {i} completed successfully.")
    except RuntimeError as e:
        if 'out of memory' in str(e):
            oom = oom+1
            #print(f"Step {i} skipped due to OOM error.")
        raise 
    seed = seed + 1
print('oom:',oom,file = file)




