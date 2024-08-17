import torch as tc
import graph_generate as gg
import numpy as np
import contract_exact as ce
import time
import EntropyBP as EBP
import EntropyFTNMP as EFT
import get_regions_new as grn
import TNBP
import nx_test as nxt



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
    - boundaries (dict): Dictionary where keys are nodes and values are boundary sets.
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
    cavity,converged,edgelist2,step = TNBP.TNBP_3_sat_interaction_new(G_fac,Nv,boundaries,clauses,tendencies,direct_products,max_item,region_info,region_bound,damping_factor,device)
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
        _,step_BP_05 = EBP.EntropyBP(clauses,tendencies,0.5)
        t6 = time.time()
        t.append(t6-t5)
        step.append(step_BP_05)

        print('\n','BP_damping = 0:')
        t9 = time.time()
        _,step_BP_0 = EBP.EntropyBP(clauses,tendencies,0)
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




