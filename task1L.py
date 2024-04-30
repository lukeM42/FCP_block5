import numpy as np
import matplotlib.pyplot as plt
import random
alpha = 0.1
external = 0.0



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
    #print(p, "p")

    critical_value = random.random()
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
  #  plt.show()


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
        #print('Step:', frame, end='\r')
        im = ax.imshow(population, interpolation='none', cmap='RdPu_r')
        plot_ising(im, population)


if __name__ == "__main__":
    population = np.random.choice([-1, 1], size=(100, 100))
    #print(population)
    ising_main(population, alpha, external)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')
    plot_ising(im, population)
   # print(population, "after")


