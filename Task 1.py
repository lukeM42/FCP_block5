import numpy as np

import matplotlib.pyplot as plt
import random


def calculate_agreement(population, row, col, external=0.0, alpha = 1.0):
    '''
    	This function should return the extent to which a cell agrees with its neighbours.
    	Inputs: population (numpy array)
    			row (int)
    			col (int)
    			external (float)
    	Returns:
    			change_in_agreement (float)
    	'''

    num_rows, num_cols = population.shape # retrieves the number of rows on columns in given population array
    # Calculate current agreement
    if row == num_rows: # wraps around to top of array if on the bottom row
        lower_neighbour_agreement = ((population[0, col]) * (population[row, col]))
    else:
        lower_neighbour_agreement = ((population[row % num_rows, col]) * (population[row, col]))
    if row == 0: # wraps around to bottom of array if on the top row
        upper_neighbour_agreement = (population[row % num_rows, col]) * (population[row, col])
    else:
        upper_neighbour_agreement = (population[row - 1, col]) * (population[row, col])

    if col == 0: # wraps around to right-most column if on left-most column
        left_neighbour_agreement = (population[row, col % num_cols]) * (population[row, col])
    else:
        left_neighbour_agreement = (population[row, col - 1]) * (population[row, col])
    if col == num_cols: # wraps around to left-most column if on right-most column
        right_neighbour_agreement = (population[row, 0]) * (population[row, col])
    else:
        right_neighbour_agreement = (population[row, col % num_cols]) * (population[row, col])
    sum_of_agreements = upper_neighbour_agreement + lower_neighbour_agreement + left_neighbour_agreement + right_neighbour_agreement
    agreement = sum_of_agreements + (external * population[row, col]) # this adds the value of external if applicable

    return agreement


def ising_step(population, external=0.0, alpha=1.0):
    '''
    This function will perform a single update of the Ising model
    Inputs: population (numpy array)
          external (float) - optional - the magnitude of any external "pull" on opinion
    '''

    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)

    p=np.exp(-(calculate_agreement(population,row,col,external=0,alpha=0))/alpha)
    critical_value = random.random()
    agreement = calculate_agreement(population, row, col)
    if critical_value < p or agreement < 0:
        population[row, col] *= -1



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


def ising_main(population, alpha=None, external=None):
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


#population = np.random.randint(-1, 1 ,(3,3))
population = np.ones((3,3))

print(population)

print(population.shape)
ising_main(population, 0.01, 0)


