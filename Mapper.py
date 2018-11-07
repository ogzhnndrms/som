"""
This script is written for mapping and naming classes of the neurons.
"""

from SOM import SOM
import numpy as np
import enum


class MotionTypes(enum.Enum):
    """ 
    Creating enumerations using class
    """
    working_at_computer = 0
    standing_up_walking_and_going_up_and_down_stairs = 1
    standing = 2
    walking = 3
    going_up_down_stairs = 4
    walking_and_talking_with_someone = 5
    talking_while_standing = 6


def map_vects():
    """
    Maps each input vector to the relevant neuron in the SOM
    grid.
    'input_vectors' should be an iterable of 1-D NumPy arrays with
    dimensionality as provided during initialization of this SOM.
    Returns a list of 1-D NumPy arrays containing (row, column)
    info for each input vector(in the same order), corresponding
    to mapped neuron.

    Use this function if you have already trained SOM (Need weights).
    """

    # Read weights.
    weights = []
    with open("weights.txt", "r") as weights_file:
        weights = map(lambda x: [float(el) for el in x.split(
            ",")], weights_file.read().splitlines())

    if weights is None:
        raise ValueError("No weights, please firstly train your SOM.")

    # Read validation data
    validation_Data = open('Validation Data/validation_file.csv')

    if validation_Data is None:
        raise ValueError("No validation data.")

    # Create a list that contains weights and classes
    weights_and_claim_array = []
    for weight in weights:
        # Fill the claim array with weight and claims about this weight.
        weights_and_claim_array.append([weight, [0, 0, 0, 0, 0, 0, 0]])

    for line in validation_Data:
        # Getline and remove \n char from it then split the line.
        vector = [int(element) for element in line.split(",")]
        type_of_motion = vector[3]
        del vector[3]
        # Find closest neuron.
        min_index = min([i for i in range(len(weights_and_claim_array))],
                        key=lambda x: np.linalg.norm([a_i - b_i for a_i, b_i in zip(vector, weights[x])]))
        weights_and_claim_array[min_index][1][type_of_motion - 1] += 1

    will_return = []
    temp = []
    counter = 6
    for i in range(0, 7):
        for weight_and_claim in weights_and_claim_array:
            temp.append(weight_and_claim[1][counter])
        index = temp.index(max(temp))
        will_return.append([weights_and_claim_array[index][0], counter])
        weights_and_claim_array[index][1] = [-1, -1, -1, -1, -1, -1, -1]
        counter -= 1
        temp[:] = []
    print(will_return)

    return will_return
