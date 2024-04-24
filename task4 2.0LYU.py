import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# Define a class Node to represent each node in a network
class Node:
    def __init__(self, state, index, connections=[]):
        # Initialize the node with a state, a unique index, and a list of connections
        self.state = state
        self.index = index
        self.connections = connections


# Define a class Network to represent the whole network of nodes
class Network:
    def __init__(self, nodes):
        # Initialize the network with a list of Node objects
        self.nodes = nodes

    def get_mean_degree(self):
        # Calculate the average degree of the network
        degrees = [sum(node.connections) for node in self.nodes]
        return np.mean(degrees)

    def get_clustering(self):
        # Calculate the average clustering coefficient for the network
        clustering_coeffs = []
        for node in self.nodes:
            if sum(node.connections) > 1:
                # Find indices of neighboring nodes
                neighbors = [i for i, x in enumerate(node.connections) if x == 1]
                neighbor_links = 0
                # Count links between neighbors
                for i in neighbors:
                    for j in neighbors:
                        if self.nodes[i].connections[j] == 1:
                            neighbor_links += 1
                total_possible_links = len(neighbors) * (len(neighbors) - 1) / 2
                clustering_coeffs.append(neighbor_links / total_possible_links)
            else:
                clustering_coeffs.append(0)
        return np.mean(clustering_coeffs) if clustering_coeffs else 0

    def get_path_length(self):
        # Calculate the average shortest path length in the network
        def bfs(start_index):
            # Perform breadth-first search from start_index
            queue = [start_index]
            distances = {start_index: 0}
            while queue:
                current = queue.pop(0)
                current_distance = distances[current]
                for i, connected in enumerate(self.nodes[current].connections):
                    if connected == 1 and i not in distances:
                        distances[i] = current_distance + 1
                        queue.append(i)
            return distances

        total_distance = 0
        count = 0
        for i in range(len(self.nodes)):
            distances = bfs(i)
            for key in distances:
                if key != i:
                    total_distance += distances[key]
                    count += 1
        return total_distance / count if count > 0 else 0

    def draw(self):
        # Visualize the network using matplotlib and networkx
        G = nx.Graph()
        for node in self.nodes:
            G.add_node(node.index)
            for i, connected in enumerate(node.connections):
                if connected == 1 and not G.has_edge(node.index, i):
                    G.add_edge(node.index, i)
        pos = nx.circular_layout(G)
        nx.draw(G, pos, node_color='orange', edge_color='blue', with_labels=True)
        plt.show()


def test_networks():
    # Ring network creation
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number - 1) % num_nodes] = 1
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)
    print("Testing ring network")
    network.draw()  # Draw the ring network
    assert (network.get_mean_degree() == 2), network.get_mean_degree()
    assert (network.get_clustering() == 0), network.get_clustering()
    assert (network.get_path_length() == 2.777777777777778), network.get_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)
    print("Testing one-sided network")
    network.draw()
    assert (network.get_mean_degree() == 1), network.get_mean_degree()
    assert (network.get_clustering() == 0), network.get_clustering()
    assert (network.get_path_length() == 5), network.get_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)
    print("Testing fully connected network")
    network.draw()
    assert (network.get_mean_degree() == num_nodes - 1), network.get_mean_degree()
    assert (network.get_clustering() == 1), network.get_clustering()
    assert (network.get_path_length() == 1), network.get_path_length()

    print("All tests passed")


# Check if this script is the main program and execute test function
if __name__ == "__main__":
    test_networks()
