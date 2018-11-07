""" This module is created for movement recognition using SOM.

    Oguzhan DURMUS oguzhann.durmus@gmail.com
"""
import sys
import tensorflow as tf
import numpy as np


class SOM(object):
    """" 2 Dimensional SOM (Self Organizing Map) """

    def __init__(self, row_count, column_count, alpha=0.1, sigma=None):
        """ Constructor

        Initializes all necessary components of the TensorFlow Graph.

        Args:

            row_count: number of rows
            column count: number of columns
                Multiplication of these numbers will give the size of graph
                which will be created in tensoflow
            iteration_count: number of iterartion for training
                In my model dimension of the input is 3
            alpha: a number denoting the initial time(iteration no)-based learning rate.
                Default value is 0.9
            sigma: the the initial neighbourhood value,
                denoting the radius of influence of the BMU while training.
                By default, its taken to be half of max(row_count, column_count).

        """
        # Defining class variables.
        self._row_count = row_count
        self._column_count = column_count
        self._alpha = alpha
        self._dimension = 3

        if sigma is None:
            self._sigma = max(self._row_count, self._column_count) / 2.0
        else:
            self._sigma = sigma
        # Assigning number of iteration
        self._number_of_iterations = 100

        # Create tensorflow graph
        self._graph = tf.Graph()

        # Graph creation
        # Tensorflow graph is used for composing some functionlities.
        with self._graph.as_default():

            # Initialize weight vectors randomly.
            self._weight_vectors = tf.Variable(
                tf.random_normal([self._row_count * self._column_count, self._dimension]))

            # Matrix of size [self._row_count * self._column_count, 2] for
            # SOM grid locations of neurons.
            self._location_vects = tf.constant(
                np.array(list(self._neuron_locations(self._row_count, self._column_count))))

            # The training vector
            # The shape means, my input vector has got .. numbers of elements
            self._vector_input = tf.placeholder("float", [self._dimension])
            # Iteration number
            # A float value, it will be fed into the computation.
            self._iter_input = tf.placeholder("float")

            # CONSTRUCT TRAINING OP PIECE BY PIECE
            # Only the final, 'root' training op needs to be assigned as
            # an attribute to self, since all the rest will be executed
            # automatically during training

            # To compute the BMU (Best Matching Unit) of given a vector.
            # Calculate euclidian distance and return
            # index of the neuron which gives the least value.
            # argmin and reduce_sum is deprecated. It must be changed.
            # The argmin returns the index with the smallest value across axes of a tensor.
            # The reduce_sum computes the sum of elements across dimensions of a tensor.
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self._weight_vectors, tf.stack(
                    [self._vector_input for i in range(self._row_count * self._column_count)])), 2), 1)),
                0)

            # Find the index of BMU
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, 2]))),
                                 [2])

            # To compute the alpha and sigma values based on iteration
            # number
            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input,
                                                       self._number_of_iterations))
            alpha_op = tf.multiply(self._alpha, learning_rate_op)
            sigma_op = tf.multiply(self._sigma, learning_rate_op)

            # Construct the op that will generate a vector with learning
            # rates for all neurons, based on iteration number and location
            # wrt BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                self._location_vects, tf.stack(
                    [bmu_loc for i in range(self._row_count * self._column_count)])), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(sigma_op, 2))))
            learning_rate_op = tf.multiply(alpha_op, neighbourhood_func)

            # Finally, the op that will use learning_rate_op to update
            # the weightage vectors of all neurons based on a particular
            # input
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [self._dimension])
                for i in range(self._row_count * self._column_count)])
            weightage_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self._vector_input for i in range(self._row_count * self._column_count)]),
                            self._weight_vectors))
            new_weights_op = tf.add(self._weight_vectors,
                                    weightage_delta)
            self._training_op = tf.assign(self._weight_vectors,
                                          new_weights_op)

            # INITIALIZE SESSION
            self._sess = tf.Session()

            # INITIALIZE VARIABLES
            init_op = tf.initialize_all_variables()
            self._sess.run(init_op)

    def _neuron_locations(self, m, n):
        """
        Yields one by one the 1-D locations of the individual neurons
        in the SOM.
        """
        # Nested iterations over both dimensions
        # to generate all 2-D locations in the map
        # yield returns generator
        for i in range(m):
            yield np.array([i, 1])

    def train(self, input_vectors):
        """
        Traning the SOM. Use inintially random vectors.

        """

        # Training iterations
        for iter_no in range(self._number_of_iterations):
            # Train with each vector one by one.
            # Create a generator and use just data not index.
            for _, input_vect in input_vectors.iterrows():
                # Substitiution and do one step operation of graph.

                self._sess.run(self._training_op,
                               feed_dict={self._vector_input: input_vect,
                                          self._iter_input: iter_no})

        # Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for i in range(self._row_count)]
        self._weights = list(self._sess.run(self._weight_vectors))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weights[i])
        self._centroid_grid = centroid_grid

        # Set true for trained
        self._trained = True

    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid

    def map_vects(self, input_vectors):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vectors' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """

        if not self._trained:
            raise ValueError("SOM not trained yet")

        to_return = []
        for _, vect in input_vectors.iterrows():
            min_index = min([i for i in range(len(self._weights))],
                            key=lambda x: np.linalg.norm(vect -
                                                         self._weights[x]))
            to_return.append(self._locations[min_index])

        return to_return
