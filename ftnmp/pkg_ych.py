from copy import deepcopy
import networkx as nx
import numpy as np




def cut(G,nodes = False):
    G_loop = deepcopy(G)
    if nodes != False:
        for i in nodes:
            if i in G_loop:
                if len(G_loop[i]) == 0:
                    G_loop.remove_node(i)
                elif len(G_loop[i]) == 1:
                    j = list(G_loop[i])[0]
                    G_loop.remove_node(i)
                    while len(G_loop[j]) == 1:
                        k = list(G_loop[j])[0]
                        G_loop.remove_node(j)
                        j = k
    else:
        nodes = list(G_loop.nodes())
        for i in nodes:
            if i in G_loop:
                if len(G_loop[i]) == 0:
                    G_loop.remove_node(i)
                elif len(G_loop[i]) == 1:
                    j = list(G_loop[i])[0]
                    G_loop.remove_node(i)
                    while len(G_loop[j]) == 1:
                        k = list(G_loop[j])[0]
                        G_loop.remove_node(j)
                        j = k
    return G_loop

def Region_generator(G,e,R):
    """Generate a subgraph Region(R) from the given edge e on the corresponding graph G, 
    which satisfies that the shortest loop cross the Region and G\Region is longer than R (cross means share more than 1 edge here).
    
    Parameters
    ----------
    G : nx.Graph
        The complete graph.
    e : tuple of int
        The beginning edge, it should be an element of list(G.edges()).
    R : int

    Returns
    -------
    Region : nx.Graph
        The subgraph.
    """
    Region = nx.Graph()
    G_environment = deepcopy(G)
    G_environment.remove_edge(e[0],e[1])
    Region.add_edge(e[0],e[1])
    boundary = list(e)
    edge_new = True
    l = 2

    while edge_new:
        edge_new = False

        new_path = []

        for i in range(1,l):
            for j in range(i):
                n1 = boundary[i]
                n2 = boundary[j]
                if nx.has_path(G_environment,n1,n2):
                    shortest_paths_out = list(nx.all_shortest_paths(G_environment, n1, n2))
                    l_shortest_paths_in = nx.shortest_path_length(Region, n1, n2)
                    if len(shortest_paths_out[0]) + l_shortest_paths_in <= R+1:
                        edge_new = True
                        new_path += shortest_paths_out
                                    
        outnode = set()
        for path in new_path:
            outnode.update(path)
            for node_id in range(len(path)-1):
                if not Region.has_edge(path[node_id],path[node_id+1]):
                    Region.add_edge(path[node_id],path[node_id+1])
                    G_environment.remove_edge(path[node_id],path[node_id+1])

        if edge_new:
            boundary = []
            for node in outnode: 
                if len(Region[node])<len(G[node]):
                    boundary.append(node)
        l = len(boundary)
        
        if l < 2:
            break
        
    return Region

def get_Regions(G,R,N):
    G = cut(G)
    G_remain = deepcopy(G)
    Regions = []
    edges = list(G.edges())
    for e in edges:
        if G_remain.has_edge(e[0],e[1]):
            g = Region_generator(G_remain,e,R)
            G_remain.remove_edges_from(list(g.edges())) 
            if len(g) > 2:
                Regions.append(devide_region(g,R,N))
            G_remain = cut(G_remain,list(g))
    return Regions

def devide_region(g,R,N):
    if len(g) <= N: #小于设置范围没有再划分的必要 
        return [g]
    else:
        gg = [g]
        g_remain = deepcopy(g)

        for e in list(g.edges()):
            if not g_remain.has_edge(e[0],e[1]):
                continue
            subg,g_remain = subregion_generator(g_remain,e,R,N)
            if len(subg)>0:
                gg.append(subg)
                g_remain = cut(g_remain,list(subg))
                if len(g_remain)<=N:
                    break
        if len(g_remain)>0:
            gg.append(g_remain)
        return gg

def subregion_generator(g,e,R,N):
    gtemp = deepcopy(g)
    region = nx.Graph()
    region.add_edge(e[0],e[1])
    gtemp.remove_edge(e[0],e[1])
    edge_new = True

    while edge_new:
        boundary = []
        for node in list(region):
            if len(region[node])<len(g[node]):
                boundary.append(node)

        edge_new = False
        nomore = False

        for r in range(3,R+1):
            l = len(boundary)
            if l < 2:
                break
            for i in range(1,l):
                for j in range(i):
                    n1 = boundary[i]
                    n2 = boundary[j]
                    if nx.has_path(gtemp,n1,n2):
                        path_out = list(nx.all_shortest_paths(gtemp,n1,n2))[0]
                        l_path_in = nx.shortest_path_length(region,n1,n2)
                        if len(path_out) + l_path_in <= r+1:
                            edge_new = True
                            if len(path_out)+len(region) > N+2:
                                nomore = True
                                continue
                            for node_id in range(len(path_out)-1):
                                nomore = False
                                region.add_edge(path_out[node_id],path_out[node_id+1])
                                gtemp.remove_edge(path_out[node_id],path_out[node_id+1])
                            break
                if edge_new: 
                    break
            if edge_new: 
                break 
        
        if nomore or len(region)==N: 
            break
    crossnodes = []
    for node in list(region):
        l = len(g[node])
        if l>2 and l>len(region[node]): 
            crossnodes.append(node)


    left_edges = nx.Graph()
    l = len(crossnodes)
    if l>1:
        for i in range(1,l):
            for j in range(i):
                n1 = crossnodes[i]
                n2 = crossnodes[j]
                if n1 in left_edges and n2 in left_edges and nx.has_path(left_edges,n1,n2): 
                    continue
                if nx.has_path(gtemp,n1,n2):
                    l1 = nx.shortest_path_length(gtemp,n1,n2)
                    path_in = nx.shortest_path(region,n1,n2)
                    l2 = len(path_in)-1
                    if l1+l2 > R: 
                        continue
                    for node_id in range(l2):
                        left_edges.add_edge(path_in[node_id],path_in[node_id+1])


    gtemp.add_edges_from(list(left_edges.edges()))

    return region,gtemp
