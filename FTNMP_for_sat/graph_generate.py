import numpy as np
import random
import networkx as nx
import subprocess
import json
import copy


def G_generator_fac(clauses):
    G = nx.Graph()    #前n个是qubit，后面是clause
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
    nodes = list(range(max_item))
    for node in nodes:
            if node not in list(G.nodes()):
                G.add_node(node) 

    return G, max_item

def generate_graph(clauses,fac_list):
    edges = []
    max_node = 0
    for clause_id in range(len(clauses)):
        clause = clauses[clause_id]
        qubit2 = fac_list[clause_id]
        max_node = max(max_node, qubit2)
        for qubit1_id in range(len(clause)):
            qubit1 = clause[qubit1_id]
            max_node = max(max_node, qubit1)
            edges.append((qubit1, qubit2))
    return edges, max_node + 1

def neighborhood_grow(G,G_neighborhood0,G_environment,R):
   
    stop = 0
    turn = 0
    G_neighborhood = copy.deepcopy(G_neighborhood0)
    G_neighborhoods = []
    new_nodes = []
    new_edges = []
    boundaries = []
    while stop == 0:
        stop = 1
        turn += 1
        new_nodes_per_turn = []
        new_edges_per_turn = []
        boundary = []
        for node in list(G_neighborhood.nodes()):
            for noden in G.neighbors(node):
                if not G_neighborhood.has_edge(node,noden):#邻居里没有这条边，就把这个点算在边界里
                    boundary.append(node)
                    break
        if turn != 1:
            boundaries.append(boundary)#边界上所有的点
        l_b = len(boundary)
        for index1 in range(l_b):
            for index2 in range(index1+1,l_b):
                end1 = boundary[index1]
                end2 = boundary[index2]
                if nx.has_path(G_environment, end1, end2) == True:
                    shortest_paths = list(nx.all_shortest_paths(G_environment, end1, end2))
                else:
                    shortest_paths = []
                while len(shortest_paths) != 0 and len(shortest_paths[0]) <= R+1 :
                    stop = 0
                    for shortest_path in shortest_paths:
                        for node_id in range(len(shortest_path)-1):
                            if not G_neighborhood.has_edge(shortest_path[node_id],shortest_path[node_id+1]):
                                if not G_neighborhood.has_node(shortest_path[node_id]):
                                    new_nodes_per_turn.append(shortest_path[node_id])
                                if not G_neighborhood.has_node(shortest_path[node_id + 1]):
                                    new_nodes_per_turn.append(shortest_path[node_id + 1])
                                new_edges_per_turn.append((shortest_path[node_id],shortest_path[node_id+1]))
                                G_neighborhood.add_edge(shortest_path[node_id],shortest_path[node_id+1])
                                G_environment.remove_edge(shortest_path[node_id],shortest_path[node_id+1])
                    if nx.has_path(G_environment, end1, end2) == True:
                        shortest_paths = list(nx.all_shortest_paths(G_environment, end1, end2))
                    else:
                        shortest_paths = []
        new_nodes.append(new_nodes_per_turn)
        new_edges.append(new_edges_per_turn)
        G_neighborhoods.append(copy.deepcopy(G_neighborhood))    
    return G_neighborhoods,new_nodes,new_edges,boundaries,G_environment,turn

def Ni_generator_sat_fac(G,clauses,center_node,R):
    G_neighborhood = nx.Graph()
    G_environment = copy.deepcopy(G)
    Ni_v = []
    direct_e = []
    for direct_neighbor_of_i in G.neighbors(center_node):
        Ni_v.append(direct_neighbor_of_i)
        direct_e.append((center_node,direct_neighbor_of_i))
    G_neighborhood.add_nodes_from(Ni_v)
    G_neighborhood.add_edges_from(direct_e)
    G_environment.remove_edges_from(direct_e)
    for r in range(1,R+1):
        G_neighborhoods,_,_,_,G_environment,_ = neighborhood_grow(G,G_neighborhood,G_environment,r)
        G_neighborhood = G_neighborhoods[-1]
    for node in list(G_neighborhood.nodes()):
        if node not in Ni_v:
            Ni_v.append(node)
    return Ni_v

def boundaries_generation(G_fac,clauses,R):
    Nv = []
    size_of_neighborhood = []
    boundaries = []

    for i in range(G_fac.number_of_nodes()):
        Ni_v_fac=Ni_generator_sat_fac(G_fac, clauses, i, R)
        #print('Ne_fac',Ni_v_fac)
        Nv.append(Ni_v_fac)
        size_of_neighborhood.append(len(Ni_v_fac))
        #print(size_of_neighborhood)
        boundary = []
        for node in Ni_v_fac:
            for noden in G_fac.neighbors(node):
                if noden not in Ni_v_fac:
                    boundary.append(node)
                    break
        boundaries.append(boundary)
    #print('Nv',Nv)
    #print('boundaries',boundaries)  
    return Nv,boundaries


def triangle_long_graph_free(turns,n_trangel,n_long):
    #首先生成一个树图
    clauses = [[2,1,0]]
    tendency0 = list(np.random.randint(0, 2, 3)*2-1)
    tendencies = [tendency0]
    rand = np.random.rand(turns)
    for turn in rand:
        u = np.random.randint(0,len(clauses))
        v = np.random.randint(0,len(clauses[u]))
        clause = [clauses[u][v],max(max(clauses))+1,max(max(clauses))+2]
        clauses.append(sorted(clause,reverse=True))
        tendency = np.random.randint(0, 2, 3)
        tendency = list(tendency * 2 - 1)
        tendencies.append(tendency) 
    max_item = max(max(clauses))
    clauses_copy = copy.deepcopy(clauses)
    #选择要加上结构的点
    #n_trange_list= np.random.randint(0,max_item,n_trangel)
    n_list = np.random.randint(0,max_item,n_trangel+n_long)
    n_trange_list = [n_list[tl] for tl in range(n_trangel)]
    n_long_list = [n_list[ll] for ll in range(n_trangel,len(n_list))]
    add_clauses = []
    devide = []
    devide_bound = []
    devide_node = []
    max_fin = max_item+3*n_trangel+len(clauses)+5*n_long

     ##要加的结构为圈长为6的嵌入结构
    for add_long in n_long_list:
        clauses_change = [clause for clause in clauses if add_long in clause]
        random.shuffle(clauses_change)

        node_communal01 = max_item+1
        node_communal02 = max_item+2
        node_communal03 = max_item+3
        node_single01 = max_item+4
        node_single02 = max_item+5

        clauses.append([node_communal02,node_communal01,add_long])
        add_clauses.append([node_communal02,node_communal01,add_long])
        clauses.append([node_single01,node_communal03,node_communal02])
        add_clauses.append([node_single01,node_communal03,node_communal02])
        clauses.append([node_single02,node_communal03,node_communal01])
        add_clauses.append([node_single02,node_communal03,node_communal01])
        for t in range(3):
            tendency = np.random.randint(0, 2, 3)
            tendency = list(tendency * 2 - 1)
            tendencies.append(tendency)

        #开始安排变化
        for clause_id in range(1,len(clauses_change)):
            p = random.randint(0,1)
            if p == 0:
                change_node =  node_single01 
            else:
                change_node =  node_single02 
            clausec = clauses_change[clause_id]
            add_node_id = clausec.index(add_long)
            cl_id = clauses.index(clausec)
            cl_id_copy = -100
            cl_id_add = -100
            if clausec in clauses_copy:
                cl_id_copy = clauses_copy.index(clausec)
            if clausec in add_clauses:
                cl_id_add = add_clauses.index(clausec)
                for devide_id in range(len(devide)):
                    dev = devide[devide_id]
                    f = len(set(dev).intersection(set(clausec)))
                    if f ==2:
                        t = devide_bound[devide_id].index(add_long)#找到划分边缘中新加点的位置
                        devide_bound[devide_id][t] = change_node
            clausec[add_node_id] =change_node
            clauses[cl_id] = sorted(clausec,reverse=True)
            if cl_id_copy >=0:
                clauses_copy[cl_id_copy] = sorted(clausec,reverse=True)
            if cl_id_add >=0:
                add_clauses[cl_id_add] = sorted(clausec,reverse=True)    
        devide.append([node_communal03,node_communal02,node_communal01,max_fin+1,max_fin+2,max_fin+3])
        max_fin = max_fin+3
        devide_bound.append([node_single02,node_single01,add_long])
        devide_node.append(node_communal03)
        devide_node.append(node_communal02)
        devide_node.append(node_communal01)
        max_item = max(max(clauses))





    #要加的结构为圈长为4的嵌入结构
    for add_node in n_trange_list:
        clauses_change = [clause for clause in clauses if add_node in clause]
        random.shuffle(clauses_change)
        '''
        if len(devide_bound)!=0:
            devide_bound_changes = [bound for bound in devide_bound if add_node in bound]
        #if len(add_clauses)!=0:
            #add_clauses_change = [add for add in add_clauses if add_node in add]
        '''
        #print('clauses_change',clauses_change)
        clauses.append([max_item+2,max_item+1,add_node])
        add_clauses.append([max_item+2,max_item+1,add_node])
        clauses.append([max_item+3,max_item+2,max_item+1])
        add_clauses.append([max_item+3,max_item+2,max_item+1])
        for clause_id in range(1,len(clauses_change)):
            clausec = clauses_change[clause_id]
            add_node_id = clausec.index(add_node)
            cl_id = clauses.index(clausec)
            cl_id_copy = -100
            cl_id_add = -100
            if clausec in clauses_copy:
                cl_id_copy = clauses_copy.index(clausec)
            if clausec in add_clauses:
                cl_id_add = add_clauses.index(clausec)
                
                #根据子句找出哪个划分区域：
                for devide_id in range(len(devide)):
                    dev = devide[devide_id]
                    f = len(set(dev).intersection(set(clausec)))
                    if f ==2:
                        t = devide_bound[devide_id].index(add_node)#找到划分边缘中新加点的位置
                        devide_bound[devide_id][t] = max_item+3


            clausec[add_node_id] = max_item+3
            clauses[cl_id] = sorted(clausec,reverse=True)
            if cl_id_copy >=0:
                clauses_copy[cl_id_copy] = sorted(clausec,reverse=True)
            if cl_id_add >=0:
                add_clauses[cl_id_add] = sorted(clausec,reverse=True)
            
            #print('clauses',clauses)
            #print('clausecopy',clauses_copy)

        devide.append([max_item+2,max_item+1,max_fin+1,max_fin+2])
        devide_bound.append([max_item+3,add_node])
        max_fin = max_fin+2
        devide_node.append(max_item+2)
        devide_node.append(max_item+1)
        tendency = np.random.randint(0, 2, 3)
        tendency = list(tendency * 2 - 1)
        tendencies.append(tendency) 
        tendency = np.random.randint(0, 2, 3)
        tendency = list(tendency * 2 - 1)
        tendencies.append(tendency) 
        max_item = max(max(clauses))
    #n_long_list= np.random.randint(0,max_item,n_long)





   

    
    for fc in add_clauses:
        devide_node.append(clauses.index(fc)+max_item+1)   
    for c in clauses_copy:
        devide.append([clauses.index(c)+max_item+1])   
    for clause in clauses_copy:
        devide_bound.append(clause) 
    return clauses,tendencies,n_trange_list,add_clauses,devide,devide_bound,devide_node



