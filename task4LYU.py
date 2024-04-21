import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, value):
        """
        Initialize a graph node.

        """
        self.id = None
        self.value = value
        self.neighbors = []

    def connect(self, other):
        """
        Connect two nodes.

        """
        if other not in self.neighbors:
            self.neighbors.append(other)
            other.neighbors.append(self)


class Graph:
    def __init__(self):
        """
        Initialize the graph object, initially without any nodes.
        """
        self.nodes = []

    def add_node(self, value):
        """
        Add a new node to the graph.

        """
        node = Node(value)
        node.id = len(self.nodes)
        self.nodes.append(node)
        return node

    def average_degree(self):
        """
        Calculate the average degree of all nodes in the graph.

        """
        if len(self.nodes) == 0:
            return 0
        return np.mean([len(node.neighbors) for node in self.nodes])

    def bfs_path_lengths(self, start):
        """
        Use breadth-first search to calculate the path lengths from the start node to all other nodes in the graph.

        """
        queue = [start]
        visited = {start: 0}
        while queue:
            current_node = queue.pop(0)
            current_distance = visited[current_node]
            for neighbor in current_node.neighbors:
                if neighbor not in visited:
                    visited[neighbor] = current_distance + 1
                    queue.append(neighbor)
        return visited

    def create_random_network(self, size, prob):
        """
        Create a random network, connecting each pair of nodes with a given probability.

        """
        self.nodes = [self.add_node(np.random.random()) for _ in range(size)]
        for i, node in enumerate(self.nodes):
            for j in range(i + 1, size):
                if np.random.random() < prob:
                    node.connect(self.nodes[j])

    def create_ring_network(self, size):
        """
        Create a ring network, connecting each node with its two direct neighbors.

        """
        self.nodes = [self.add_node(np.random.random()) for _ in range(size)]
        for i in range(size):
            next_node = self.nodes[(i + 1) % size]
            previous_node = self.nodes[(i - 1 + size) % size]
            self.nodes[i].connect(next_node)
            if previous_node != next_node:
                self.nodes[i].connect(previous_node)

    def create_small_world_network(self, size, prob=0.2):
        """
        Create a small world network by starting with a ring network and then reconnecting each edge with a given probability.

        """
        self.create_ring_network(size)
        for i, node in enumerate(self.nodes):
            for j, potential_new_neighbor in enumerate(self.nodes):
                if j != i and np.random.random() < prob:
                    old_neighbor = node.neighbors.pop(0)
                    old_neighbor.neighbors.remove(node)
                    node.connect(potential_new_neighbor)

    def plot_network(self):
        """
        Visualize the network.
        """
        fig, ax = plt.subplots()
        pos = {node.id: (np.cos(2 * np.pi * node.id / len(self.nodes)), np.sin(2 * np.pi * node.id / len(self.nodes)))
               for node in self.nodes}
        for node in self.nodes:
            ax.scatter(*pos[node.id], color='red')
            for neighbor in node.neighbors:
                ax.plot([pos[node.id][0], pos[neighbor.id][0]], [pos[node.id][1], pos[neighbor.id][1]], 'black')

        ax.set_aspect('equal')
        plt.axis('off')
        plt.show()


def test_networks(graph):
    graph.create_ring_network(10)
    print("Testing ring network")
    assert graph.average_degree() == 2, f"Expected degree 2, got {graph.average_degree()}"

    graph.nodes = [graph.add_node(i) for i in range(10)]
    for i in range(9):
        graph.nodes[i].connect(graph.nodes[i + 1])
    print("Testing one-sided network")
    assert graph.average_degree() == 1.8, f"Expected degree 1.8, got {graph.average_degree()}"
    graph.nodes = [graph.add_node(i) for i in range(10)]
    for i in range(10):
        for j in range(i + 1, 10):
            graph.nodes[i].connect(graph.nodes[j])
    print("Testing fully connected network")
    assert graph.average_degree() == 9, f"Expected degree 9, got {graph.average_degree()}"
    print("All tests passed")


def main():
    """
    Main function to create, test, and visualize networks.
    """
    graph = Graph()
    graph.create_random_network(10, 0.5)
    test_networks(graph)
    graph.plot_network()

    graph.create_small_world_network(10, 0.3)
    test_networks(graph)
    graph.plot_network()


if __name__ == "__main__":
    main()
