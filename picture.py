import networkx as nx
import matplotlib.pyplot as plt


def draw_custom_graph(G, points_list, threshold, lenth):
    """
    Draw a graph with customized node styles based on a threshold and group points with different fill colors.
    
    Parameters
    ----------
    G : nx.Graph
        The input graph to be drawn.
    points_list : list of list of int
        A 2D list where each sublist contains node indices to be filled with the same color.
    threshold : int
        The threshold to differentiate between different node shapes and colors.
    lenth: int
        The longest list element that needs to be colored
        
    """
    points_list = [points_list[id] for id in range(lenth)]
    # Create a color map for the groups in the points_list
    color_map = plt.get_cmap('Accent')
    
    # Prepare node colors and shapes
    node_colors = {}
    node_shapes = {}
    fill_colors = {}
    
    # Assign colors to nodes based on their groups in points_list
    for i, group in enumerate(points_list):
        fill_color = color_map(i % 10)
        for node in group:
            fill_colors[node] = fill_color
    
    # Assign node shapes and edge color
    for node in G.nodes():
        if G.degree(node)!=0:
            if node < threshold:
                node_shapes[node] = ('o', 'black')  # Deep blue hollow circle
            else:
                node_shapes[node] = ('s', 'black')  # Maroon hollow square
            
        # Check if the node should be filled with a specific color
        node_colors[node] = fill_colors.get(node, 'none')  # 'none' means no fill
    
    # Draw the graph
    pos = nx.kamada_kawai_layout(G)
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='black', width=0.3)
    
    # Draw nodes with different shapes and colors
    for node, (shape, edgecolor) in node_shapes.items():
        nx.draw_networkx_nodes(
            G, pos, nodelist=[node], node_color=node_colors[node],
            node_shape=shape, edgecolors=edgecolor, linewidths=0.3, 
            node_size=20 
        )
    
    # Draw node labels
    #nx.draw_networkx_labels(G, pos, font_color='black', font_size=10)
    
    # Show the plot
    plt.axis('off')
    plt.show()
    #plt.savefig('fig.jpg')

