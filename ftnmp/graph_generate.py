import numpy as np
import random
import networkx as nx
import copy
import matplotlib.pyplot as plt



def random_3_sat_graph(n_qubit,alpha):
    nodes = list(range(n_qubit))
    n_clauses = int(n_qubit * alpha)
    clauses = []
    tendencies = []
    i = 0
    #for clause_id in range(n_clauses):
    while i < n_clauses:
        clause = random.sample(nodes, 3)
        clause =sorted(clause,reverse=True)
        if clause not in clauses:
            #clauses.append(sorted(clause,reverse=True))
            clauses.append(clause)
            tendency = np.random.randint(0, 2, 3)
            tendency = list(tendency * 2 - 1)
            tendencies.append(tendency)
            i= i+1 

    return clauses,tendencies

def random_k_sat_graph(n_qubit,alpha,k):
    nodes = list(range(n_qubit))
    n_clauses = int(n_qubit * alpha)
    clauses = []
    tendencies = []
            
    for clause_id in range(n_clauses):
        clause = random.sample(nodes, k)
        if clause not in clauses:
            clauses.append(sorted(clause,reverse=True))
            tendency = np.random.randint(0, 2, k)
            tendency = list(tendency * 2 - 1)
            tendencies.append(tendency) 

    return clauses,tendencies

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

def double_random_generate(n_glo,alpha_glo,n_cluster,alpha_range,min_cluster,max_cluster,seed):
    np.random.seed(seed) 
    random.seed(seed)
    clauses,tendencies = random_3_sat_graph(n_glo,alpha_glo)
    cluster_sizes = []
    alphas = []
    for i in range(n_cluster):
        cluster_size = np.random.randint(min_cluster,max_cluster)
        alpha = random.uniform(alpha_range[0], alpha_range[1])
        cluster_sizes.append(cluster_size)
        alphas.append(alpha)
    print(cluster_sizes)
    print(alphas)
    nodes = list(range(max(max(clauses))))
    for cs_id in range(len(cluster_sizes)):
        cs = cluster_sizes[cs_id]
        alpha = alphas[cs_id]
        cluster = random.sample(nodes, cs)
        print(cluster)
        n_subclauses = int(cs * alpha)
        for ns in range(n_subclauses):
            clause = random.sample(cluster, 3)
            if cluster not in clauses:
                clauses.append(sorted(clause,reverse=True))
                tendency_cluster = np.random.randint(0, 2, 3)
                tendency_cluster = list(tendency_cluster * 2 - 1)
                tendencies.append(tendency_cluster) 
    return clauses,tendencies
