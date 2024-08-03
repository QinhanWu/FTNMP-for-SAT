from copy import deepcopy
import networkx as nx
import numpy as np
import ftnmp.graph_generate as gg

'''
调整内容:用来编写最新的region生成函数
先实现二次划分，不考虑公共边更新的问题
'''

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
            for node in outnode: # 由于我们考虑的是圈的长度，因此如果一个点在一轮中没有进入outnode，也就意味着这个点不属于任何一个和上一轮区域相交的小环，那么新添加了点之后也不会属于了（因为该点到新的点必然经过上一轮判定过的点，所属的圈也必然包含途径的点）
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
        # print(e)
        if G_remain.has_edge(e[0],e[1]):
            # print(e)
            # t = time.time()
            g = Region_generator(G_remain,e,R)
            # print('Region',time.time()-t)
            # print(len(g))
            G_remain.remove_edges_from(list(g.edges())) # 不管怎样，g的所有边都可以从G_remain上移除了
            # print(list(g))
            if len(g) > 2:# 在一条边的基础上找到了其他区域
                # t = time.time()
                Regions.append(devide_region(g,R,N))
                # print('subRegion',time.time()-t)
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
            # print(list(subg),list(g_remain))
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

    # 先找到新的subregion
    while edge_new:
        boundary = []
        for node in list(region):
            if len(region[node])<len(g[node]):
                boundary.append(node)
        # print(boundary)

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
                            # print(path_out,list(region))
                            if len(path_out)+len(region) > N+2: # 添加新边会超范围，就跳过
                                nomore = True
                                continue
                            for node_id in range(len(path_out)-1):
                                nomore = False
                                region.add_edge(path_out[node_id],path_out[node_id+1])
                                gtemp.remove_edge(path_out[node_id],path_out[node_id+1])
                            # print('getmp',list(gtemp))
                            break
                if edge_new: # 一次只能添加一条新的边
                    break
            if edge_new: # 一次只能添加一条新的边
                break 
        
        if nomore or len(region)==N: # 当超过上限或恰好抵达上限时break
            break

    # 需要保留subregion和g_remain的有意义的公共边
    crossnodes = []
    for node in list(region):
        l = len(g[node])
        if l>2 and l>len(region[node]): # l=2的的话肯定不可能是公共点喽
            crossnodes.append(node)

    # print('gtemp',list(gtemp),'edge',list(gtemp.edges()),len(list(gtemp.edges())))

    left_edges = nx.Graph()
    l = len(crossnodes)
    if l>1:
        for i in range(1,l):
            for j in range(i):
                n1 = crossnodes[i]
                n2 = crossnodes[j]
                if n1 in left_edges and n2 in left_edges and nx.has_path(left_edges,n1,n2): # 当两个点有路径相连时就先放过他们吧，在当前处理的情况来说，大概就是对的？
                    continue
                if nx.has_path(gtemp,n1,n2):
                    l1 = nx.shortest_path_length(gtemp,n1,n2)
                    path_in = nx.shortest_path(region,n1,n2)
                    l2 = len(path_in)-1
                    if l1+l2 > R: # 注意length给出的不是点的数目，而是边的数目，所以l1不会多1
                        continue
                    # print(path_in)
                    for node_id in range(l2):
                        left_edges.add_edge(path_in[node_id],path_in[node_id+1])

    # print(left_edges.edges(),'gtemp',list(gtemp),'edge',list(gtemp.edges()),len(list(gtemp.edges())))

    gtemp.add_edges_from(list(left_edges.edges()))

    return region,gtemp

def safe_swap(T,i,j = None,forward = True): # dir 为0是换过来，1是换回去
    '''
    turn a0a1a2a3...ai...aj...b0b1b2b3...bi...bj... to ai aj bi bj a0a1a2a3...b0b1b2b3...
    and back
    '''
    if j!=None and j<i:
        [i,j] = [j,i]

    o = len(list(T.shape))//2

    if forward:
        T = np.expand_dims(T,axis=0)
        T = np.expand_dims(T,axis=0)
        if j == None:
            T = np.swapaxes(T,0,i+2)
            T = np.swapaxes(T,1,i+o+2)
            T = np.squeeze(T,i+o+2)
            T = np.squeeze(T,i+2)
        else:
            T = np.expand_dims(T,axis=0)
            T = np.expand_dims(T,axis=0)
            T = np.swapaxes(T,0,i+4)
            T = np.swapaxes(T,1,j+4)
            T = np.swapaxes(T,2,i+o+4)
            T = np.swapaxes(T,3,j+o+4)
            T = np.squeeze(T,j+o+4)
            T = np.squeeze(T,i+o+4)
            T = np.squeeze(T,j+4)
            T = np.squeeze(T,i+4)
    else:
        if j == None:
            T = np.expand_dims(T,axis=(i+2))
            T = np.expand_dims(T,axis=(i+o+2))
            T = np.swapaxes(T,0,i+2)
            T = np.swapaxes(T,1,i+o+2)
        else:
            T = np.expand_dims(T,axis=(i+4))
            T = np.expand_dims(T,axis=(j+4))
            T = np.expand_dims(T,axis=(i+o+4))
            T = np.expand_dims(T,axis=(j+o+4))
            T = np.swapaxes(T,0,i+4)
            T = np.swapaxes(T,1,j+4)
            T = np.swapaxes(T,2,i+o+4)
            T = np.swapaxes(T,3,j+o+4)
            T = np.squeeze(T,0)
            T = np.squeeze(T,0)
        T = np.squeeze(T,0)
        T = np.squeeze(T,0)

    return T

def getneigh(G,nodes):
    neighs = []
    for i in nodes:
        neigh = []
        for j in G[i]:
            if j not in nodes:
                neigh.append(j)
        neighs.append(neigh)
    return neighs

def generate_graph_wang(n, degree=3, seed=0, G_type='rrg', branch=2, height=2):
    n_nodes = n
    if G_type == 'rrg':
        G = nx.random_regular_graph(degree, n_nodes, seed=seed)
    elif G_type == 'erg':
        G = nx.erdos_renyi_graph(n, degree / n, seed=seed)
    elif G_type == 'tree':
        G = nx.balanced_tree(branch, height)
    return G

def generate_graph_clique_wang(n,base_degree,seed,num3,maxk):
    """生成全局rrg、局部clique的graph
    
    Parameters
    ----------
    base_degree: int
        RRG的节点度数
    num3: int
        为保证每种大小的clique边数一致,设定大小为k的clique的数目=int(6*num3/(k*(k-1)))
    maxk: int 
        最大clique的节点数
    """
    G = generate_graph_wang(n, degree=base_degree, seed=0, G_type='rrg', random_weights=False, branch=0, height=0)
    for k in range(2,maxk+1):
        numk = int(6*num3/(k*(k-1)))
        np.random.seed(seed+k)
        for i_clique in range(numk):
            #在所有节点中任选k个点建立k阶clique
            vertices_choice = np.random.randint(low=0,high=n,size=(k))
            for i1 in range(len(vertices_choice)):
                for i2 in range(i1+1,len(vertices_choice)):
                    v1 = vertices_choice[i1]
                    v2 = vertices_choice[i2]
                    if v2 not in list(G.neighbors(v1)):
                        G.add_edge(v1,v2)

    #如果生成的图有多个连通分支，则通过在两个分支之间增加两条连边来使它们连通
    while nx.number_connected_components(G) != 1:
        cc1 = list(list(nx.connected_components(G))[0])
        cc2 = list(list(nx.connected_components(G))[1])
        v1 = cc1[0]
        v2 = cc2[0]
        if v2 not in list(G.neighbors(v1)):
            G.add_edge(v1,v2)
        if len(cc1) > 1 and len(cc2) > 1:
            v1 = cc1[-1]
            v2 = cc2[-1]
            if v2 not in list(G.neighbors(v1)):
                G.add_edge(v1,v2)
    
    return G

def get_boundaries(LG):
    l = len(LG)-1
    boundaries = [[None for _ in range(l)] for _ in range(l)]
    for i in range(1,l):
        g1 = set(LG[i+1])
        for j in range(i):
            g2 = set(LG[j+1])
            boundary = list(g1.intersection(g2))
            if len(boundary)>0:
                boundaries[i][j] = boundary
                boundaries[j][i] = boundary
    return boundaries

def safe_inv(v):
    inv = np.zeros_like(v)
    for i in range(len(v)):
        if v[i]>0:
            inv[i] = 1./v[i]
        else:
            inv[i] = 0.
    return inv

