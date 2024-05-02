import random

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
		# calculates the mean of the mean path length for each node and returns it

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
		'''
		creates a ring network with the amount of nodes N and range 1
		:param N:
		:param neighbour_range:
		:return:
		'''
		self.nodes = []
		for index in range(N):
			connections = [0 for index in range(N)]
			for i in range(-neighbour_range,neighbour_range+1):
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

	def plot(self,ax):


		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)

			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
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
	# gets the neighbour nodes that are connected to the main node
	for neighbour1 in neighbours:
		for neighbour2 in neighbours:
			if not(neighbour1 == neighbour2) and neighbour1.connections[neighbour2.index] == 1:
				# if they aren't the same node and are connected, increment amount of connections
				connections += 1
	return connections / 2
	# returns the amount of connections. / 2 is needed as each connection is counted twice


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

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''

def calculate_agreement(population, row, col, external=0.0, alpha=1.0):
    '''
    This function returns the extent to which a cell agrees with its neighbors.
    Inputs: population (numpy array)
            row (int)
            col (int)
            external (float)
    Returns:
            change_in_agreement (float)
    '''
    num_rows, num_cols = population.shape

    # Calculate current agreement
    lower_neighbour_agreement = population[(row + 1) % num_rows, col] * population[row, col]
    upper_neighbour_agreement = population[(row - 1) % num_rows, col] * population[row, col]
    left_neighbour_agreement = population[row, (col - 1) % num_cols] * population[row, col]
    right_neighbour_agreement = population[row, (col + 1) % num_cols] * population[row, col]

    sum_of_agreements = upper_neighbour_agreement + lower_neighbour_agreement + \
                        left_neighbour_agreement + right_neighbour_agreement
    agreement = sum_of_agreements + (external * population[row, col])

    return agreement

def ising_step(population, alpha, external):
    '''
    This function will perform a single update of the Ising model
    Inputs: population (numpy array)
          external (float) - optional - the magnitude of any external "pull" on opinion
    '''

    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)
    agreement = calculate_agreement(population, row, col)
    p = np.exp(-(agreement) / alpha)

    critical_value = np.random.random()
    if critical_value < p or agreement < 0:
        population[row, col] *= -1

    return population


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
    assert (calculate_agreement(population, 1, 1, 10) == -6), "Test 9"
    assert (calculate_agreement(population, 1, 1, -10) == 14), "Test 10"

    print("Tests passed")

def ising_main(population, alpha, external):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, alpha, external)
        print('Step:', frame, end='\r')
        im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

        plot_ising(im, population)

'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''
def update_opinions(opinions,T,b):
	"""
	updates a list of opinions, returns the updated list
	"""
	updated_opinions = opinions
	# creates a separate array to add the new opinions onto
	for i in range(len(opinions)): #reperates for the length of how many opinions there are
		person1 = random.randint(0,len(opinions)) # picks a random person
		if np.random.rand() > 0.5:
			# uses a 50/50 chance to pick which neighbour is used
			if person1 == len(opinions)-1:
				person2 = 0
			else:
				person2 = person1 + 1
			# additional if statement is needed to prevent out of bounds error
		else:
			person2 = person1 - 1

		opinion1 = opinions[person1]
		opinion2 = opinions[person2]

		if ((opinion1 - opinion2)**2)**0.5 < T:
			# takes the magnitude of the difference of opinion and checks if it's lower than the threshold
			updated_opinions[person1] = changed_opinion(opinion1,opinion2,b)
			updated_opinions[person2] = changed_opinion(opinion2,opinion1,b)
			# updates both opinions
	return updated_opinions

def update_opinion_network(opinion_nodes,T,b):
	'''
	Args:
		opinion_nodes: A list of nodes containing each opinion
		T: Threshold value
		b: Beta value

	Returns: The updated list of nodes
	'''
	updated_opinion_nodes = opinion_nodes
	# creates a separate array to add the new opinions onto

	for i in range(len(opinion_nodes)): # iterates for the amount of people there are
		person1_index = np.random.randint(0,len(opinion_nodes))  # picks a random one
		connected_indexes = opinion_nodes[person1_index].get_neighbour_indexes()

		person2_index = np.random.randint(0,len(connected_indexes))  #picks a random one of their neighbours
		opinion1 = opinion_nodes[person1_index].value
		opinion2 = opinion_nodes[person2_index].value

		if ((opinion1 - opinion2)**2)**0.5 < T:
			# takes the magnitude of the difference of opinion and checks if its lower then the threshold
			updated_opinion_nodes[person1_index].value = changed_opinion(opinion1,opinion2,b)
			updated_opinion_nodes[person2_index].value = changed_opinion(opinion2,opinion1,b)
			# updates both opinions
	return updated_opinion_nodes

def changed_opinion(opinionA, opinionB, b):
	'''
	Args:
		opinionA: opinion of a person
		opinionB: opinion of their neighbour
		b: Beta value
	Returns: the updated opinion of that person
	'''
	return opinionA + b * (opinionB - opinionA)


def defuant_main_network(b, T, network=None):
	'''
	Args:
		b: Beta value
		T: Threshold value
		network: network to use if one is supplied
	'''

	opinions = [node.value for node in network.nodes] # to track the values of the nodes across that timestep
	mean = [np.mean(opinions)]
	opinion_max = [max(opinions)]
	opinions_min = [min(opinions)]
	opinion_range = [max(opinions) - min(opinions)]
	fig = plt.figure()
	ax = fig.add_subplot(111) # declares the animation plot outside the for loop so it redraws on same plot

	for t in range(1, 101):  # iterates through the 100 time steps
		network.nodes = update_opinion_network(network.nodes, T, b) # updates the opinions
		opinions = [node.value for node in network.nodes]
		mean.append(np.mean(opinions))
		opinion_range.append(max(opinions) - min(opinions))
		opinion_max.append(max(opinions))
		opinions_min.append(min(opinions))
		ax.cla() # clears the previous plot to re-draw the new one
		ax.set_axis_off()
		network.plot(ax) # plots the current network at that timestep
		plt.pause(0.03) # displays the plot for 0.03 seconds


	plt.show()  # calling show here prevents the plot from immediately going after the last timestep
	plt.title("Mean, max and min opinion over timesteps")
	plt.ylabel("Opinion")
	plt.xlabel("Timestep")
	plt.grid()
	plt.plot([t for t in range(len(mean))], mean)
	plt.plot([t for t in range(len(opinion_max))], opinion_max)
	plt.plot([t for t in range(len(opinions_min))], opinions_min)
	plt.legend(["Mean", "Max", "Min"])
	plt.show()

	plt.title("Range max and min of opinions over timesteps")
	plt.ylabel("Opinion range")
	plt.xlabel("Timestep")
	plt.grid()
	plt.plot([t for t in range(len(opinion_range))], opinion_range)
	plt.show()

def defuant_main(b, T):
	'''
	Args:
		b: Beta value
		T: Threshold value
	'''
	opinions = np.random.rand(100)  # creates a list of 100 random floats between 0 and 1
	fig = plt.figure()
	graph1 = fig.add_subplot(121)  # adds the plot for the histogram
	plt.xlabel('Opinions')
	plt.xlim([0,1])
	graph2 = fig.add_subplot(122)  # adds the plot for the opinions against the timestep
	graph2.scatter([0 for i in range(len(opinions))],opinions, c='red')  # plots the initial set of opinions
	for t in range(1, 101):  # iterates through the 100 time steps
		opinions = update_opinions(opinions, T, b)  # calls function to update all the opinions each time
		graph2.scatter([t for i in range(len(opinions))], opinions, c='red')  # plots the set of opinions for that time step
	graph1.hist(opinions, bins=[i / 10 for i in range(11)])  #creates the histogram with 11 bins going from 0 to 1 in increments of 0.1
	plt.ylabel('Opinions')
	plt.ylim([0, 1])
	plt.show()


def test_defuant():
	"""
	Tests the Defaunt model functions
	"""
	print("Testing Defuant model")
	opinions = [0.1,0.9]
	updated = [0.1,0.9]
	# tests threshold check
	assert (update_opinions(opinions, 0.5, 0.2) == updated), "Test 1"
	# tests if the opinions are changed correctly
	assert (np.round(changed_opinion(0.2,0.4,0.2),4) == 0.24), "Test 2"
	assert (np.round(changed_opinion(0, 1, 0.5), 4) == 0.5), "Test 3"
	print("All tests passed")

'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main():
	network = Network()
	#You should write some code for handling flags here
	parser = argparse.ArgumentParser()

	# Ising model flags
	parser.add_argument("-ising_model", action='store_true')
	parser.add_argument("-external", nargs=1, default=0, type=int)
	parser.add_argument("-alpha", nargs=1, default=1, type=int)
	parser.add_argument("-test_ising", action='store_true')

	# Defuant model flags
	parser.add_argument("-defuant", action='store_true')
	parser.add_argument("-beta", nargs=1, default=0.2, type=float)
	parser.add_argument("-threshold", nargs=1, default=0.2, type=float)
	parser.add_argument("-test_defuant",action='store_true')

	# Networks flags
	parser.add_argument("-network", nargs=1, default=-1, type=int)
	parser.add_argument("-test_network", action='store_true')
	parser.add_argument("-probability", nargs=1, default=0.3, type=float)

	# Ring and small world network flags
	parser.add_argument("-ring_network", nargs=1, type=int, default=-1)
	parser.add_argument("-small_world", nargs=1, type=int, default=-1)
	parser.add_argument("-re_wire", nargs=1, type=float, default=0.2)

	# Opinion formation on networks flags
	parser.add_argument("-use_network", nargs=1, type=int, default=-1)

	args = parser.parse_args()

	# Ising model flag handling
	if args.ising_model:
		if type(args.alpha) == list:
			args.alpha = args.alpha[0]
		if type(args.external) == list:
			args.external = args.external[0]
		population = np.random.choice([-1, 1], size=(100, 100))
		ising_main(population, args.alpha, args.external)

	if args.test_ising:
		test_ising()

	# Defuant model flag handling
	if args.test_defuant:
		test_defuant()
	if args.defuant:
		if type(args.beta) == list:
			args.beta = args.beta[0]  # as passed in arguments are stored in lists, they therefore must be removed
		if type(args.threshold) == list:  # from the list before they can be used
			args.threshold = args.threshold[0]
		if type(args.use_network) == list:
			args.use_network = args.use_network[0]
			if type(args.re_wire) == list:
				network.make_small_world_network(args.use_network, args.re_wire[0])
				defuant_main_network(args.beta, args.threshold, network)
			else:
				network.make_small_world_network(args.use_network, args.re_wire)
				defuant_main_network(args.beta, args.threshold, network)
		else:
			defuant_main(args.beta, args.threshold)

	# Network flag handling
	if args.test_network:
		test_networks()

	if type(args.network) == list:
		args.network = args.network[0]
		if type(args.probability) == list:
			args.probability = args.probability[0]
		network.make_random_network(args.network,args.probability)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()
		network.plot(ax)
		plt.show()
		# creates a random network of size N, then plots it
		print("Random network mean degree:", network.get_mean_degree())
		print("Random network mean clustering coefficient:", network.get_clustering())
		print("Random network mean path length:", network.get_path_length())
		# outputs details about that network

	# Ring and Small world flag handling
	if type(args.ring_network) == list:
		args.ring_network = args.ring_network[0]
		network.make_ring_network(args.ring_network, 1)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()
		network.plot(ax)
		plt.show()
		# creates a ring network of size N and a neighbour range of 1, then plots it
		print("Ring network mean degree:", network.get_mean_degree())
		print("Ring network mean clustering coefficient:", network.get_clustering())
		print("Ring network mean path length:", network.get_path_length())
		# outputs details about that network

	if type(args.small_world) == list:
		args.small_world = args.small_world[0]
		if type(args.re_wire) == list:
			args.re_wire = args.re_wire[0]
		network.make_small_world_network(args.small_world,args.re_wire)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()
		network.plot(ax)
		plt.show()
		# creates a small world network of size N with a rewiring probability of re_wire
		print("Small world network mean degree:", network.get_mean_degree())
		print("Small world network mean clustering coefficient:", network.get_clustering())
		print("Small world network mean path length:", network.get_path_length())
		# outputs details about that network


if __name__=="__main__":
	main()
