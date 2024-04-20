import numpy as np
import matplotlib.pyplot as plt


class GraphNode:
    def __init__(self, id, value):
        """
        Initialise the graph node.

        """
        self.id = id
        self.value = value

        # Used to store neighbouring nodes directly connected to this node
        self.neighbors = []

    def connect(self, other):
        """
       Connects two nodes.

        """
        if other not in self.neighbors:
            self.neighbors.append(other)  # Add other node to self node's neighbour list.

            other.neighbors.append(self)  # Other.neighbors.append(self) adds self node to other node's neighbour list


class Graph:
    def __init__(self):
        """
        Initialise the graph object, there are initially no nodes in the graph.

        """
        self.nodes = []

    def add_node(self, value):
        """
        Add a new node to the graph.

        """
        node = GraphNode(len(self.nodes), value)
        self.nodes.append(node)
        return node

    def average_degree(self):
        """
       Computes the average degree of all nodes in the graph (the number of connections per node).

        """
        return np.mean([len(node.neighbors) for node in self.nodes])

    def clustering_coefficient(self):
        """
        Calculate the average aggregation factor of the graph.

        """
        coeffs = []
        for node in self.nodes:
            if len(node.neighbors) > 1:
                actual_links = 0
                for i in node.neighbors:
                    for j in node.neighbors:
                        if i != j and i in j.neighbors:
                            actual_links += 1
                coeffs.append(actual_links / (len(node.neighbors) * (len(node.neighbors) - 1)))
            else:
                coeffs.append(0)
        return np.mean(coeffs)

    def average_path_length(self):
        """
        Calculate the average path length between all pairs of nodes in the graph.

        """
        lengths = []
        for start in self.nodes:
            path_lengths = self.bfs_path_lengths(start)
            lengths.extend(path_lengths.values())
        return np.mean(lengths)

    def bfs_path_lengths(self, start):
        """
        Computes the length of the path from the start node to all other nodes in the graph using breadth-first search.

        """
        queue = [start]
        visited = {start: 0}
        while queue:
            node = queue.pop(0)
            current_distance = visited[node]
            for neighbor in node.neighbors:
                if neighbor not in visited:
                    visited[neighbor] = current_distance + 1
                    queue.append(neighbor)
        return visited

    def create_random_network(self, size, prob):
        """
        Create a random network where each pair of nodes is connected to each other with a certain probability.

        """
        self.nodes = [self.add_node(np.random.random()) for _ in range(size)]
        for i, node in enumerate(self.nodes):
            for j in range(i + 1, size):
                if np.random.random() < prob:
                    node.connect(self.nodes[j])

    def create_ring_network(self, size):
        """
        Create a ring network where each node is connected to its left and right nodes.

        """
        self.nodes = [self.add_node(np.random.random()) for _ in range(size)]
        for i, node in enumerate(self.nodes):
            next_node = self.nodes[(i + 1) % size]
            node.connect(next_node)

    def plot_network(self):
        """
        Visualisation of the network structure.
        """
        fig, ax = plt.subplots()
        pos = {node.id: (np.cos(2 * np.pi * node.id / len(self.nodes)),
                         np.sin(2 * np.pi * node.id / len(self.nodes))) for node in self.nodes}
        for node in self.nodes:
            ax.scatter(*pos[node.id], color='red')
            for neighbor in node.neighbors:
                ax.plot([pos[node.id][0], pos[neighbor.id][0]],
                        [pos[node.id][1], pos[neighbor.id][1]], 'black')

        ax.set_aspect('equal')
        plt.axis('off')
        plt.show()


def test_networks(graph):
    """
    Test the characteristics of the network and output the results.

    """
    print("Average Degree:", graph.average_degree())
    print("Clustering Coefficient:", graph.clustering_coefficient())
    print("Average Path Length:", graph.average_path_length())


def main():
    """
    Main function, create network, test and visualise
    """
    graph = Graph()
    graph.create_random_network(10, 0.5)
    test_networks(graph)
    graph.plot_network()


if __name__ == "__main__":
    main()
