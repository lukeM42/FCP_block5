class Node:
    def __init__(self, value, index, connections):
        self.value = value
        self.index = index
        self.connections = connections

    def get_neighbour_indexes(self):
        return [i for i, x in enumerate(self.connections) if x == 1]


class Network:
    def __init__(self, nodes=None):
        self.nodes = nodes if nodes is not None else []


def update_opinion_network(opinion_nodes, T, b):
    updated_opinion_nodes = list(opinion_nodes)  # Ensure a deep copy if necessary
    for person1_index in range(len(opinion_nodes)):
        connected_indexes = opinion_nodes[person1_index].get_neighbour_indexes()
        for person2_index in connected_indexes:
            opinion1 = opinion_nodes[person1_index].value
            opinion2 = opinion_nodes[person2_index].value
            if abs(opinion1 - opinion2) < T:
                updated_opinion_nodes[person1_index].value = opinion1 + b * (opinion2 - opinion1)
                updated_opinion_nodes[person2_index].value = opinion2 + b * (opinion1 - opinion2)
    return updated_opinion_nodes


def defuant_main(args, network):
    if network:
        b = args.beta
        T = args.threshold
        opinions = [node.value for node in network.nodes]
        mean_opinions = [np.mean(opinions)]

        fig, ax = plt.subplots()
        ax.set_xlabel('Time step')
        ax.set_ylabel('Mean Opinion')

        for t in range(100):
            network.nodes = update_opinion_network(network.nodes, T, b)
            opinions = [node.value for node in network.nodes]
            mean_opinions.append(np.mean(opinions))

        ax.plot(range(101), mean_opinions, label='Mean Opinion over Time')
        ax.legend()
        plt.show()
    else:
        print("No network provided.")


def main():
    parser = argparse.ArgumentParser(description="Run the Deffuant model on a network.")
    parser.add_argument("-beta", type=float, default=0.1, help="Influence factor (b) for the Deffuant model.")
    parser.add_argument("-threshold", type=float, default=0.5, help="Threshold (T) for the Deffuant model.")
    args = parser.parse_args()

    nodes = [Node(0.1 * i, i, [1 if j == i - 1 or j == i + 1 else 0 for j in range(10)]) for i in range(10)]
    network = Network(nodes)

    defuant_main(args, network)
