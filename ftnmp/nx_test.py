import networkx as nx
from itertools import combinations
import torch as tc

def find_intersections(devides):
    """Find all intersections between pairs of lists in devides."""
    intersections = []
    for (i, list1), (j, list2) in combinations(enumerate(devides), 2):
        intersection = set(list1).intersection(set(list2))
        if intersection:
            intersections.append((i, j, list(intersection)))
    return intersections

def find_edges(G_fac, intersections):
    """Find edges in G_fac that are formed by nodes in intersections."""
    edges = []
    for i, j, intersection in intersections:
        for u, v in combinations(intersection, 2):
            if G_fac.has_edge(u, v):
                edges.append((i, j, tuple(sorted((u, v)))))
    return edges

def find_direct_products(max_item,intersections):
    direct_products = []
    for i, j, intersection in intersections:
        if max(intersection)<=max_item:
                 direct_products.append((i, j, intersection))
    return  direct_products


def find_else_situations(max_item,intersections):
    else_situations = []
    for i, j, intersection in intersections:
        if len(intersection)> 2 :#and max(intersection)>max_item:
                 else_situations.append((i, j, intersection))
    return  else_situations


def cluster_edges_generate(G_fac, devides, devide_bound):
    # Step 1: Find all intersections
    intersections = find_intersections(devides)
    
    # Step 2: Find edges in G_fac based on intersections
    edges = find_edges(G_fac, intersections)


    
    return intersections, edges


def cluster_edges_generate_new(G_fac,max_item, devides, devide_bound):
    # Step 1: Find all intersections
    intersections = find_intersections(devides)
    
    # Step 2: Find edges in G_fac based on intersections
    edges = find_edges(G_fac, intersections)

    direct_products = find_direct_products(max_item,intersections)

    else_situations = find_else_situations(max_item,intersections)
    
    return intersections, edges, direct_products, else_situations





