
import graph_generate as gg
import numpy as np
import nx_test as nxt
import get_regions_new as grn
from tqdm import tqdm




def devides_ansyinfo(G_fac,clauses,max_item,R_region,R_subregion):
    devides,subdevides,region_info,single_list = grn.get_regions_z(G_fac,clauses,max_item,R_region,R_subregion)
    devide_bound,degree_node = grn.devides_others(G_fac,devides)
    intersection_info, edges_info = nxt.cluster_edges_generate(G_fac,devides,devide_bound)
    
    region_info_len = []
    for r in region_info:
       region_info_len.append(len(r))

    
    rate = max(region_info_len)/(1+max_item+len(clauses))   

    len_ii = []
    len_ii1 = []
    len_ii2 = []
    len_else = []

    check = []
    len_len_else = [0]
    for ii in intersection_info:
        len_ii.append(len(ii[2]))
        
        
        if len(ii[2]) >2:
            len_else.append(ii)
            len_len_else.append(len(ii[2]))
        if len(ii[2]) ==2:
            if max(ii[2]) <= max_item:
                len_ii1.append(ii)
                len_len_else.append(len(ii[2]))
            else:
                len_ii2.append(ii)
        

    print('max_region_nodes---mrn/total---max_Overlap_nodes---mOn_number')
    info = [max(region_info_len),rate,max(len_len_else),len(len_ii1)+len(len_else)]
    print(info)
    
  

    return  info



seed = 0
info4s = []
info6s = []
file = open('NEIGHBOR.txt','w')
for i in tqdm(range(50)):
    nodes = 450
    alpha =0.5
    n_cl = 6
    alpha_local=[2.5,3.2]
    max_cl = 12
    min_cl = 8
    clauses,tendencies = gg.double_random_generate(nodes,alpha,n_cl,alpha_local,min_cl,max_cl,seed)
    if i ==0:
        print('nodes:',nodes,'alpha_grobal:',alpha,'number of cluster:',n_cl,'alpha_local:',alpha_local,'Maximum points:',max_cl,'Minimum points:',min_cl,file = file)
    G_fac,max_item =gg.G_generator_fac(clauses)
    nodes = list(range(max_item+1))
    print(f"Step {i}  begin:")
    
    R_region = 4
    info4 = devides_ansyinfo(G_fac,clauses,max_item,R_region,100)
    info4s.append(info4)
    R_region = 6
    info6 = devides_ansyinfo(G_fac,clauses,max_item,R_region,100)
    info6s.append(info6)
    seed = seed+1
    print(info4,file = file)
    print(info6,file = file)

info4_m = np.mean(np.array(info4s),axis=0)
info6_m = np.mean(np.array(info6s),axis=0)
print('max_region_nodes---mrn/total---max_Overlap_nodes---mOn_number',file=file)
print('R = 4/5',info4_m,file=file)
print('R = 6/7',info6_m,file=file)

