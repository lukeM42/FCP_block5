import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Node:

	def __init__(self, value, number, connections=None, parent = None):

		self.index = number
		self.connections = connections
		self.value = value
		self.parent = parent

	def get_neighbour_indexes(self):
		'''
		Returns the indexes of the nodes connected itself
		:return:
		'''
		# if there is a connected node at that index, add it to the list
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
		return len(self.queue)==0


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
		return total_degree / len(self.nodes) # calculates and then returns the mean

	def get_path_length(self):
		'''
		Calculates the mean value of a nodes path length to a connected node
		It will then calculate and return the mean of the mean values for each node.
		:return:
		'''
		mean_path_length = 0
		for node1 in self.nodes:
			total_path_length = 0
			amount_reachable = 0
			for node2 in self.nodes:
				if not(node1 == node2) and breadth_first_search(node1, node2, self.nodes):
					route = [node.value for node in breadth_first_search(node1, node2, self.nodes)]
					path_length = len(route) - 1
					total_path_length += path_length
					amount_reachable += 1
			if amount_reachable:
				mean_path_length += total_path_length / amount_reachable
		return mean_path_length / len(self.nodes)


	def get_clustering(self):
		total_clustering_coeff = 0
		for node in self.nodes:
			n = sum(node.connections)
			if (n * (n - 1) / 2):
				possible_connections = (n * (n - 1) / 2)
				actual_connections = amount_neighbour_connections(node,self.nodes)
				total_clustering_coeff += actual_connections / possible_connections
		return total_clustering_coeff / len(self.nodes)


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
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1

	def make_ring_network(self, N, neighbour_range=1):
		self.nodes = []
		for i in range(N):
			connections = [0 for i in range(N)]
			connections[i-1] = 1
			if i+1 < N:
				connections[i+1] = 1
			else:
				connections[0] = 1
			self.nodes.append(Node(np.random.random(),i,connections))


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
							node.connections[neighbour_index-N] = 1
							self.nodes[neighbour_index-N].connections[index] = 1



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

			circle = plt.Circle((node_x, node_y), 0.6 * num_nodes, color=cm.hot(node.value))
			#circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, color=cm.hot(node.value))
			ax.add_patch(circle)

			for neighbour_index in range(i + 1, num_nodes):
				if node.connections[neighbour_index]:
					neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
					neighbour_x = network_radius * np.cos(neighbour_angle)
					neighbour_y = network_radius * np.sin(neighbour_angle)

					ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

def breadth_first_search(start_node,end_node,nodes):
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
		current_node = queue.pop()
		if current_node == end_node:
			break
		for neighbour_index in current_node.get_neighbour_indexes():
			if nodes[neighbour_index] not in visited:
				queue.push(nodes[neighbour_index])
				visited.append(nodes[neighbour_index])
				nodes[neighbour_index].parent = current_node
	current_node = end_node
	start_node.parent = None
	path = []
	while current_node.parent:
		path.append(current_node)
		current_node = current_node.parent
	path.append(current_node)
	path = [node for node in path[::-1]] # reverses the path order so it goes from start to finish
	if start_node.value == path[0].value and end_node.value == path[-1].value:
		# if it was able to complete the path return it
		return path
	else:
		# otherwise return False
		return False

def amount_neighbour_connections(main_node,nodes,connections = 0):
	'''
	calculates the amount of neighbour to neighbour connections a nodes neighbours have
	:param main_node: node to look at neighbours from
	:param nodes: array containing the nodes
	:return:
	'''
	neighbours = [nodes[neighbour_index] for neighbour_index in main_node.get_neighbour_indexes()]
	for neighbour1 in neighbours:
		for neighbour2 in neighbours:
			if not(neighbour1 == neighbour2) and neighbour1.connections[neighbour2.index] == 1:
				connections += 1
	return connections / 2


def test_networks():

	#Ring network
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number-1)%num_nodes] = 1
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	network.plot()
	plt.show()

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert (network.get_path_length() == 2.777777777777778), network.get_path_length()
	assert(network.get_clustering()==0), network.get_clustering()


	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	network.plot()
	plt.show()

	print("Testing one-sided network")
	assert(network.get_mean_degree()==1), network.get_mean_degree()
	assert(network.get_clustering()==0),  network.get_clustering()
	assert(network.get_path_length()==5), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	network.plot()
	plt.show()

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	#assert(network.get_clustering()==1),  network.get_clustering()
	assert(network.get_path_length()==1), network.get_path_length()

	print("All tests passed")

def main():
	#You should write some code for handling flags here
	#network = Network()
	#network.make_random_network(5, 0.3)
	#network.make_ring_network(20)
	#network.make_small_world_network(10,0.5)
	#network.plot()
	test_networks()
	#print("mean degree", network.get_mean_degree())
	#print("mean path length", network.get_mean_path_length())
	#print("mean clustering coeff", network.get_mean_clustering())
	#plt.show()





if __name__=="__main__":
	main()
