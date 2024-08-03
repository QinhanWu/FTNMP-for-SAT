import copy
import TNBP as TNBP
from collections import Counter
import itertools as it
import pkg_ych as ph
import Find as F


def merge_regions(graph, regions,clauses, split_point):
    def get_neighbors(node):
        return list(graph.neighbors(node))
    
    def is_variable_node(node):
        return node <= split_point
    
    def should_merge(region1, region2):
        # Check if two regions have intersection
        if set(region1) & set(region2) :
        
            check = 0
            for node in region2:
                if node>split_point:
               
                    nei = get_neighbors(node)
                    if set(nei)<=(set(region1) & set(region2)):
                        check = check+1
            for node in region1:
                if node>split_point:
                    
                    nei = get_neighbors(node)
                   
                    if set(nei)<=(set(region1) & set(region2)):

                        check = check+1
                        


            if (len(set(region1) & set(region2))>2 and max(set(region1) & set(region2))>split_point) or check!=0:
                return True
        
        factor_neighbors_in_other_region = {}
        for node in region1:
            if not is_variable_node(node):  # It's a factor node
                for neighbor in get_neighbors(node):
                    if neighbor not in region1:
                        if neighbor in region2:
                            factor_neighbors_in_other_region[neighbor] = factor_neighbors_in_other_region.get(neighbor, 0) + 1
                            if len(factor_neighbors_in_other_region) >= 2:
                                return True
            
        return False

    merged = True
    while merged:
        merged = False
        new_regions = []
        skip_indices = set()
        for i, region in enumerate(regions):
            if i in skip_indices:
                continue
            to_merge = set(region)
            for j, other_region in enumerate(regions):
                if j != i and j not in skip_indices:
                    
                    if should_merge(region, other_region) ==True :
                        to_merge.update(other_region)
                        if len(region)==1:
                            
                            to_merge.update(set(get_neighbors(region[0])))
                        skip_indices.add(j)
                        merged = True
            new_regions.append(list(to_merge))
        regions = new_regions
    return regions

def cheke_dangerous(G,devides,clauses,max_item):
    devides_one_dim = list(it.chain(*devides))
    for fac_node in range(max_item+1,len(clauses)+max_item+1):
        if fac_node not in devides_one_dim:
  
            nei = list(G.neighbors(fac_node))
            #check_fac = 0
            for devide_id in range(len(devides)):   
                devide = devides[devide_id]
                if set(nei) < set(devide):
                    devides[devide_id].append(fac_node)
                    devides[devide_id] = list(set(devides[devide_id]))
                if len(set(nei).intersection(set(devide)))==2:
                    print(2)
                    devides[devide_id] = devides[devide_id]+[fac_node]+nei
                    devides[devide_id] = list(set(devides[devide_id]))
    return devides

def devides_to_Nv(G_fac,devides,region_info):
    
    Nv =[]
    boundaries = []
    
    devides_one_dim = list(it.chain(*region_info))
   
    for node in sorted(list(G_fac.nodes())):
        if node not in devides_one_dim:
            nei = list(G_fac.neighbors(node))
        
            Nv.append(nei+[node])
            boundaries.append(nei)
          
        else:
           
            index =F.find_all_indices_2d(region_info, node)
           
           
            if len(index)==1:
                nei = list(G_fac.neighbors(node))
               
                Nv_ele = list(set(nei+region_info[index[0][0]]))
            
                boundary = []
                for noden in Nv_ele:
                    nein = list(G_fac.neighbors(noden))
                    Q = set(nein).difference(set(Nv_ele))
                    if len(Q) != 0:
                        boundary.append(noden)
                
            else:
               
                Nv_ele = list(G_fac.neighbors(node))
               
                for u in range(len(index)):

                    Nv_ele = Nv_ele + region_info[index[u][0]]
                   
                    
                Nv_ele = list(set(Nv_ele))
                boundary = []
                for noden in Nv_ele:
                    nein = list(G_fac.neighbors(noden))
                    Q = set(nein).difference(set(Nv_ele))
                    if len(Q) != 0:
                        boundary.append(noden)
    
            Nv.append(Nv_ele)
            boundaries.append(boundary)
    return Nv,boundaries

def devides_others(G_fac,devides):
    devide_bound = []
    for devide in devides:
        db = []
        for node_do in devide:
            nei = set(G_fac.neighbors(node_do)).difference(set(devide))
            if len(nei) !=0:
                db.append(node_do)
        devide_bound.append(list(set(db)))
    
    dbs = list(it.chain(*devide_bound))
    degree_node = Counter(dbs)
    return devide_bound,degree_node

def regions_compelete(G,regions,max_item):
    devides = []
    for region in regions:
        devide =  list(region)
        devide_copy = copy.deepcopy(devide)
       
        for node in devide_copy: 
         
            if node > max_item:
                nei = list(G.neighbors(node))
                devide = list(set(devide+nei))
                
        devides.append(sorted(devide))
    return devides

def get_regions_z(G,clauses,max_item,R_region,R_subregion):
    
    devides = []
    subdevides = []
    ignore_var_node=[]
    region_info = []
    regions = ph.get_Regions(G,R_region,R_subregion)
    regions = [list(re[0]) for re in regions]
    
    for region in regions:
       
        devide =  list(region)
        devide_copy = copy.deepcopy(devide)
    
        for node in devide_copy: 
            if node > max_item:
                nei = list(G.neighbors(node))
                devide = list(set(devide+nei))
               
        devides.append(sorted(devide))
      
    devides_one_dim = list(it.chain(*devides))
    devides = devides+[[i] for i in range(max_item+1,max_item+1+len(clauses)) if i not in devides_one_dim]
    devides = merge_regions(G,devides,clauses,max_item)
    devides = [devide for devide in devides if len(devide)>1]
    devides = cheke_dangerous(G,devides,clauses,max_item)
    

    ignore_var_node =[]
    for devide_id in range(len(devides)):
        devide = devides[devide_id]
        region_info_ele = []
        for noded in devide:
            if len(set(list(G.neighbors(noded)))&set(devide)) > 1: #and noded <= max_item:
                ignore_var_node.append(noded)
                region_info_ele.append(noded)
        region_info.append(region_info_ele)
    ignore_var_node = set(ignore_var_node)
    ignore_var_node = [ivn for ivn in set(ignore_var_node) if ivn<=max_item]
    devides_one_dim = list(it.chain(*devides))
    for var_node in range(max_item+1):
        if var_node not in ignore_var_node:
            nei = list(G.neighbors(var_node))
            devides.append([var_node]+nei)
    
    for fac_node in range(max_item+1,len(clauses)+max_item+1):
        if fac_node not in devides_one_dim:
            nei = list(G.neighbors(fac_node))
            devides.append([fac_node]+nei)
    return devides,subdevides,region_info,ignore_var_node



def region_info_change(G_fac,clauses,region_info,max_item,ignore_var_node):

    devides_one_dim = list(it.chain(*region_info))
    for var_node in range(max_item+1):
        if var_node not in ignore_var_node:
            region_info.append([var_node])
    for fac_node in range(max_item+1,len(clauses)+max_item+1):
    
        if fac_node not in devides_one_dim:
            region_info.append([fac_node])
    region_bound = []
    for devide in region_info:
        db = []
        for node_do in devide:
            nei = set(G_fac.neighbors(node_do)).difference(set(devide))
            if len(nei) !=0:
                db.append(node_do)
        region_bound.append(list(set(db)))
    return region_info,region_bound