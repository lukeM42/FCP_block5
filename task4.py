import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

class Node:

    def __init__(self, value, number, connections=None, parent=None):
        self.index = number
        self.connections = connections
        self.value = value
        self.parent = parent

    def get_neighbour_indexes(self):
        '''
		Returns the indexes of the nodes connected itself
		:return:
		'''
        # if there is a connected node at that index, add it to the list to return
        return [i[0] for i in enumerate(self.connections) if i[1] == 1]


class Queue:
    def __init__(self):
        self.queue = []

    def push(self, item):
        self.queue.append(item)

    def pop(self):
        if len(self.queue) < 1:
            return None
        return self.queue.pop(0)

    def is_empty(self):
        return len(self.queue) == 0


class Network:

    def __init__(self, nodes=None):

        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

    def get_mean_degree(self):
        '''
		This function calculates the mean degree value for all the nodes in a network
		A nodes degree value is equal to how many edges it has
		'''
        total_degree = 0
        for node in self.nodes:
            # iterates through each node
            total_degree += sum(node.connections)
        # this will sum the amount of connections that node has and adds it to a total
        return total_degree / len(self.nodes)  # calculates and then returns the mean

    def get_clustering(self):
        '''
		Calculates and returns the mean clustering coefficient of all the nodes
		:return:
		'''
        total_clustering_coeff = 0
        for node in self.nodes:
            # iterate through each node
            n = sum(node.connections)
            # n = number of neighbours
            if (n * (n - 1) / 2):
                # if statement required to remove possible divide by 0 error
                possible_connections = (n * (n - 1) / 2)
                # calculates amount of possible connections between neighbours
                actual_connections = amount_neighbour_connections(node, self.nodes)
                # clustering coefficient = connections between neighbours  / possible connections between neighbours
                total_clustering_coeff += actual_connections / possible_connections
        return total_clustering_coeff / len(self.nodes)

    # returns the mean clustering coefficient of all the nodes

    def get_path_length(self):
        '''
		Calculates the mean value of a nodes path length to a connected node
		It will then calculate and return the mean of the mean values for each node.
		:return:
		'''
        mean_path_length = 0
        for node1 in self.nodes:
            # iterates through each node
            total_path_length = 0
            amount_reachable = 0
            for node2 in self.nodes:
                # iterates through each node again so that it can calculate path length from node1 to node2
                if not (node1 == node2) and breadth_first_search(node1, node2, self.nodes):
                    # if the start node is different from the end node and there's a possible path between them
                    route = [node.value for node in breadth_first_search(node1, node2, self.nodes)]
                    # calculate the route from node1 to node2
                    path_length = len(route) - 1
                    total_path_length += path_length
                    amount_reachable += 1
            # as there must be a possible path between nodes, the end node must be reachable
            if amount_reachable:
                # if statement to prevent divide by 0 error
                mean_path_length += total_path_length / amount_reachable
        # calculates the mean path length of that node
        return mean_path_length / len(self.nodes)

    # caculates the mean of the mean path length for each node and returns it

    def make_random_network(self, N, connection_probability):
        '''
		This function makes a *random* network of size N.
		Each node is connected to each other node with probability p
		'''

        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index + 1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

    def make_ring_network(self, N, neighbour_range=1):
        '''
		creates a ring network with the amount of nodes N and range 1
		:param N:
		:param neighbour_range:
		:return:
		'''
        self.nodes = []
        for index in range(N):
            connections = [0 for index in range(N)]
            for i in range(-neighbour_range, neighbour_range + 1):
                if i:  # this if statement means it won't lead to it saying it's connected to itself
                    if index + i < N:
                        connections[index + i] = 1
                    else:
                        connections[index + i - N] = 1
            self.nodes.append(Node(np.random.random(), index, connections))

    def make_small_world_network(self, N, re_wire_prob=0.2):
        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index - 2, index + 3):
                if neighbour_index - index:
                    if np.random.random() < re_wire_prob:
                        random_index = int(np.random.random() * N)
                        while random_index == index and N > 1:  # N>1 needed to prevent while true loop
                            # Generates a random index until the random index isn't the index of the node itself
                            random_index = int(np.random.random() * N)
                        node.connections[random_index] = 1
                        self.nodes[random_index].connections[index] = 1
                    else:
                        if neighbour_index < N:
                            node.connections[neighbour_index] = 1
                            self.nodes[neighbour_index].connections[index] = 1
                        else:
                            node.connections[neighbour_index - N] = 1
                            self.nodes[neighbour_index - N].connections[index] = 1

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
        ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i + 1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')


def breadth_first_search(start_node, end_node, nodes):
    '''
	Completes a breadth first search from one value to another in a set of nodes, if it can't find the end node
	due to it not being reachable, it will return False, otherwise it returns the path through the nodes.
	:param start_node: The node to start the search from.
	:param end_node: The node to search for.
	:param nodes: A list containing the set of nodes
	:return:
	'''
    queue = Queue()
    queue.push(start_node)
    visited = []
    while not queue.is_empty():
        # continues whilst the queue isn't empty
        current_node = queue.pop()
        if current_node == end_node:
            break
        # exits the while loop once it has found the end node
        for neighbour_index in current_node.get_neighbour_indexes():
            # iterates through the indexes of each connected neighbour
            if nodes[neighbour_index] not in visited:
                # if it hasn't visited it already, add it to the queue to go to next
                queue.push(nodes[neighbour_index])
                visited.append(nodes[neighbour_index])
                # add it to visited now even though it's not yet been visited as it will be visited and
                # adding it now prevents the possibility of it being added to the queue multipul times
                nodes[neighbour_index].parent = current_node
        # set its parent to the current node so that it leads back to the start node through the parents
    start_node.parent = None
    path = []
    while current_node.parent:
        # whilst there is a parent to traverse to
        path.append(current_node)
        current_node = current_node.parent
    # append the current node to the path and then traverse to next potential parent
    path.append(current_node)
    path = [node for node in path[::-1]]  # reverses the path order so it goes from start to finish
    if start_node.value == path[0].value and end_node.value == path[-1].value:
        # if it was able to complete the path return it
        return path
    else:
        # otherwise return False
        return False


def amount_neighbour_connections(main_node, nodes, connections=0):
    '''
	calculates the amount of neighbour to neighbour connections a nodes neighbours have
	:param main_node: node to look at neighbours from
	:param nodes: array containing the nodes
	:return:
	'''
    neighbours = [nodes[neighbour_index] for neighbour_index in main_node.get_neighbour_indexes()]
    # gets the neighbour nodes that are connected to the main node
    for neighbour1 in neighbours:
        for neighbour2 in neighbours:
            if not (neighbour1 == neighbour2) and neighbour1.connections[neighbour2.index] == 1:
                # if they aren't the same node and are connected, increment amount of connections
                connections += 1
    return connections / 2


# returns the amount of connections. / 2 is needed as each connection is counted twice


this

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''


def calculate_agreement(population, row, col, external=0.0):
    '''
	This function should return the *change* in agreement that would result if the cell at (row, col) was to flip it's value
	Inputs: population (numpy array)
			row (int)
			col (int)
			external (float)
	Returns:
			change_in_agreement (float)
	'''

    # Your code for task 1 goes here

    return np.random * population


def ising_step(population, external=0.0):
    '''
	This function will perform a single update of the Ising model
	Inputs: population (numpy array)
			external (float) - optional - the magnitude of any external "pull" on opinion
	'''

    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)

    agreement = calculate_agreement(population, row, col, external=0.0)

    if agreement < 0:
        population[row, col] *= -1


# Your code for task 1 goes here

def plot_ising(im, population):
    '''
	This function will display a plot of the Ising model
	'''

    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.1)


def test_ising():
    '''
	This function will test the calculate_agreement function in the Ising model
	'''

    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert (calculate_agreement(population, 1, 1) == 4), "Test 1"

    population[1, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == -4), "Test 2"

    population[0, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == -2), "Test 3"

    population[1, 0] = 1.
    assert (calculate_agreement(population, 1, 1) == 0), "Test 4"

    population[2, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == 2), "Test 5"

    population[1, 2] = 1.
    assert (calculate_agreement(population, 1, 1) == 4), "Test 6"

    "Testing external pull"
    population = -np.ones((3, 3))
    assert (calculate_agreement(population, 1, 1, 1) == 3), "Test 7"
    assert (calculate_agreement(population, 1, 1, -1) == 5), "Test 8"
    assert (calculate_agreement(population, 1, 1, 10) == 14), "Test 9"
    assert (calculate_agreement(population, 1, 1, -10) == -6), "Test 10"

    print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''

# def defuant_main():
# Your code for task 2 goes here

# def test_defuant():
# Your code for task 2 goes here


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''


def main():
    # You should write some code for handling flags here




    test_networks()

    network = Network()
    network.make_ring_network(10)
    network.plot()
    plt.show()
    print("Ring network mean degree:", network.get_mean_degree())
    print("Ring network mean clustering coefficient:", network.get_clustering())
    print("Ring network mean path length:", network.get_path_length())
    network.make_small_world_network(20, 1)
    network.plot()
    plt.show()
    print("Small world network mean degree:", network.get_mean_degree())
    print("Small world network mean clustering coefficient:", network.get_clustering())
    print("Small world network mean path length:", network.get_path_length())


if __name__ == "__main__":
    main()