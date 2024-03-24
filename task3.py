import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Node:

	def __init__(self, value, number, connections=None, parent = None):

		self.index = number
		self.connections = connections
		self.value = value
		self.parent = parent

	def get_children_indexes(self):
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

	def get_mean_path_length(self):
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




	#def get_mean_clustering(self):
		#Your code for task 3 goes here



	def make_random_network(self, N, connection_probability):
		'''
		This function makes a *random* network of size N.
		Each node is connected to each other node with probability p
		'''

		self.nodes = []
		for node_number in range(N):
			value = node_number
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))

		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1

	#def make_ring_network(self, N, neighbour_range=1):
		#Your code  for task 4 goes here

	#def make_small_world_network(self, N, re_wire_prob=0.2):
		#Your code for task 4 goes here

	def plot(self):
		'''
		Plots the network of nodes
		:return:
		'''
		fig = plt.figure()
		fig.set_facecolor('blue')##################TO_REMOVE########used so that it's easier to see nodes
		ax = fig.add_subplot(111)
		ax.set_axis_off()

		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)

			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value/10))
			ax.add_patch(circle)

			for neighbour_index in range(i+1, num_nodes):
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

		for child_node_index in current_node.get_children_indexes():
			if nodes[child_node_index] not in visited:
				queue.push(nodes[child_node_index])
				visited.append(nodes[child_node_index])
				nodes[child_node_index].parent = current_node
	current_node = end_node
	start_node.parent = None
	path = []
	while current_node.parent:
		path.append(current_node)
		current_node = current_node.parent
	path.append(current_node)
	path = [node for node in path[::-1]] # reverses the path order so it goes from start to finish
	if start_node.value == path[0].value and end_node.value == path[-1].value:
		return path
	else:
		return False

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

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert(network.get_clustering()==0), network.get_clustering()
	assert(network.get_path_length()==2.777777777777778), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

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

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	assert(network.get_clustering()==1),  network.get_clustering()
	assert(network.get_path_length()==1), network.get_path_length()

	print("All tests passed")

def main():
	#You should write some code for handling flags here
	network = Network()
	network.make_random_network(5, 0.2)
	network.plot()
	print("mean degree", network.get_mean_degree())
	print("mean path length", network.get_mean_path_length())
	plt.show()









if __name__=="__main__":
	main()
