
import numpy as np
import os
# library for training progress bar
from tqdm import tqdm_notebook as tqdm
# library used to print model summary
from prettytable import PrettyTable
from math import log, floor
import matplotlib.pyplot as plt
import pickle

class ActivationLayer(object):
    def __init__(self, name, activation_type):
        self.name = name
        self.activation_type = activation_type
        supported_activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'softmax', 'linear']
        if self.activation_type not in supported_activations:
            raise ValueError('Activation Function {} not supported'.format(self.activation_function))

    def activation_function(self, input_val):
        if self.activation_type == 'relu':
            return np.maximum(0, input_val)
        elif self.activation_type == 'leaky_relu':
            return np.where(input_val > 0, input_val, input_val * 0.01)
        elif self.activation_type == 'linear':
            return input_val
        # clipping input to avoid thrown nans on overflow  underflow
        input_val = np.clip(input_val, -500, 500)
        if self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-1 * input_val))
        elif self.activation_type == 'tanh':
            return (np.exp(2 * input_val) - 1) / (np.exp(2 * input_val) + 1)
        elif self.activation_type == 'softmax':
            return np.exp(input_val) / np.exp(input_val).sum(axis=0)

    def gradient_activation_function(self, input_val):
        output_val = np.copy(input_val)
        # just get the derivative of each function and evaluate
        if self.activation_type == 'relu':
            # Heaviside function
            output_val[input_val > 0] = 1
            output_val[input_val < 0] = 0
            output_val[input_val == 0] = 0.5
            return output_val
        elif self.activation_type == 'leaky_relu':
            output_val[input_val > 0] = 1
            output_val[input_val < 0] = 0.01
            output_val[input_val == 0] = 0.5
            return output_val
        elif self.activation_type == 'linear':
            return 1
        elif self.activation_type == 'sigmoid':
            return self.activation_function(output_val) * (1 - self.activation_function(output_val))
        elif self.activation_type == 'tanh':
            return 1 - np.power(self.activation_function(output_val), 2)

    def forward(self, input):
        return self.activation_function(input)

    def backward(self, prev_pre_activation_gradients, previous_weights):
        return np.dot(previous_weights.T, prev_pre_activation_gradients)

class PreActivationLayer(object):
    def __init__(self,name,dimensions,initialization_type='glorot'):
        self.name = name
        # is the dimension of the weights as tuple ( current layer size , input to the activation fn weight )
        self.dimensions = dimensions
        self.weights = None
        self.bias = None
        self.dw = None
        self.db = None
        self.d_preactivation = None
        # we have 3 initialization types zeros, uniform , glorot
        self.initialize_weigths(initialization_type)

    def initialize_weigths(self,initialization_type):
        self.bias = np.zeros((self.dimensions[0],1))
        if initialization_type == 'zeros':
            self.weights = np.zeros(self.dimensions)
        elif initialization_type =='normal':
            self.weights = np.random.normal(0,1,self.dimensions)
        elif initialization_type =='glorot':
            uniform_range = np.sqrt(6) / np.sqrt(np.sum(self.dimensions))
            self.weights = np.random.uniform(-1 * uniform_range ,uniform_range ,self.dimensions)
        else:
            raise ValueError('Not supported Initialization type {}'.format(initialization_type))

    def forward(self,input):
        return np.dot(self.weights, input) + self.bias

    def backward(self, prev_activation_output, prev_weight  ,gradient_activation,gradient_pre_activation):
        n_items_batch = prev_activation_output.shape[1]
        self.d_preactivation = np.multiply(gradient_activation, gradient_pre_activation)
        self.dw = (1.0/n_items_batch )*  np.dot(self.d_preactivation, prev_activation_output.T)
        self.db = (1.0/n_items_batch )* np.sum(self.d_preactivation,axis=1, keepdims=True)

class NeuralLayer(object):
    # its dimension is hidden layer size * its input size
    def __init__(self, name, dimension, parent, activation_type, initialization_type):
        self.name = name
        self.dimension = dimension
        self.pre_activation = PreActivationLayer(name + '_pre_activation', self.dimension, initialization_type)
        self.output_activation = ActivationLayer(name + '_activation', activation_type)

        self.cached_pre_activation_output_forward = None
        self.cached_activation_output_forward = None

        self.cached_preactivation_gradient = None
        self.cached_activation_gradient = None

    def forward(self, input):
        self.cached_pre_activation_output_forward = self.pre_activation.forward(input)
        self.cached_activation_output_forward = self.output_activation.forward(
            self.cached_pre_activation_output_forward)
        return self.cached_activation_output_forward

    def backward(self, prev_activation_output, prev_weight, prev_pre_activation_gradients):
        gradient_pre_activation = self.output_activation \
            .gradient_activation_function(self.cached_pre_activation_output_forward)

        self.cached_activation_gradient = self.output_activation \
            .backward(prev_pre_activation_gradients, prev_weight)

        self.pre_activation \
            .backward(prev_activation_output, prev_weight
                      , self.cached_activation_gradient, gradient_pre_activation)

        self.cached_preactivation_gradient = self.pre_activation.d_preactivation

class NN(object):
    def __init__(self, hidden_dims=(512, 128), n_hidden=2, mode='train' \
                 , datapath='datasets', model_path=None, seed=1069, min_batch_size=128 \
                 , learning_rate=0.1, hidden_activations=['sigmoid', 'sigmoid'], initialization_type='glorot'):

        # hyper parameters

        self.min_batch_size = min_batch_size
        self.hidden_dims = hidden_dims
        self.hidden_activations = hidden_activations
        self.learning_rate = learning_rate

        # seed for multiple stable runs
        self.seed = seed
        self.num_iterations = 10
        self.n_hidden = n_hidden
        self.initialization_type = initialization_type
        self.datapath = datapath
        self.model_path = model_path
        self.n_labels = None
        self.input_layer = None
        self.output_layer = None
        self.input_dim = None
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.test_x = None
        self.test_y = None
        self.hidden_layers = []

        self.validate_inputs()

        if mode == 'train':
            self.load_data()
        elif mode == 'test':
            pass  # load the model set
        else:
            raise Exception('Input mode {} not working'.format(mode))

    def validate_inputs(self):
        # input validation
        if not (self.datapath):
            raise ValueError('Please input the path of the data to start the train / test')
        if len(self.hidden_dims) != self.n_hidden:
            raise ValueError('Number of hidden unit {}\
             not equal number of items {} in hidden_dims' \
                             .format(len(self.hidden_dims), self.n_hidden))

        if self.initialization_type not in ['zeros', 'normal', 'glorot']:
            raise ValueError('Intiialization type {} not supported'.format(self.initialization_type))
        if len(self.hidden_dims) != len(self.hidden_activations):
            raise ValueError('Number of hidden unit {}\
             not equal number of items {} in hidden_activations' \
                             .format(len(self.hidden_dims), self.hidden_activations))

    def initialize_weights(self, verbose=1):
        input_size = self.input_dim
        parent = None
        self.hidden_layers = []
        for i in range(self.n_hidden):
            hidden_layer = NeuralLayer('h_' + str(i), (self.hidden_dims[i], input_size), \
                                       parent, self.hidden_activations[i], self.initialization_type)
            self.hidden_layers.append(hidden_layer)
            input_size = self.hidden_dims[i]
        self.output_layer = NeuralLayer('output', (self.n_labels, self.hidden_dims[-1]), \
                                        self.hidden_layers[-1], 'softmax', self.initialization_type)
        if verbose > 0:
            self.model_summary()

    def model_summary(self):
        model_summary_table = PrettyTable(['Layer', 'Input Shape', 'Param #'])
        model_summary_table.add_row(['Input', '(None, ' + str(self.input_dim) + ')', 0])
        total_n_params = 0
        for layer in self.hidden_layers + [self.output_layer]:
            n_params = (layer.dimension[0] * layer.dimension[1]) + layer.dimension[0]
            input_dimensions = '(' + ','.join([str(dim) for dim in reversed(layer.dimension)]) + ')'
            model_summary_table.add_row([layer.name, input_dimensions, formatting_numbers(n_params)])
            total_n_params += n_params
        model_summary_table.add_row(['', 'Total # Params', formatting_numbers(total_n_params)])
        print(model_summary_table)

    def load_data(self):
        (train, val, test) = np.load(os.path.join(self.datapath, 'mnist.pkl.npy'))
        # after checking the data it is already divided by 255
        self.train_x = train[0]
        self.train_y = train[1]
        self.val_x = val[0]
        self.val_y = val[1]
        self.test_x = test[0]
        self.test_y = test[1]
        self.input_dim = self.train_x.shape[1]
        self.n_labels = np.unique(self.train_y).shape[0]

    def forward(self, input):
        # calculating the forward pass of the input data through the network
        cached_input_val = np.transpose(input)
        for hidden_indx in range(self.n_hidden):
            cached_input_val = self.hidden_layers[hidden_indx].forward(cached_input_val)
        output = self.output_layer.forward(cached_input_val)
        return output

    def activation(self, input):
        raise NotImplementedError(
            'Didn\'t need to implement ths function as i have a class including the needed activations')

    def loss(self, true_labels, prediction):
        # eps used to avoid nans in case of 0 as log(0) is nan
        eps = 1e-15
        prediction = np.clip(prediction, eps, 1 - eps)
        cost = -1 * (np.eye(self.n_labels)[true_labels] * np.log(prediction)).sum(axis=1)
        return np.average(cost)

    def accuracy(self, true_labels, prediction):
        prediction = np.argmax(prediction, axis=1)
        return np.mean(prediction == true_labels) * 100

    def softmax(self, input):
        activation = ActivationLayer('softmax', 'softmax')
        return activation.activation_function(input)

    def backward(self, train_x, train_y):
        n_items_batch = len(train_y)
        # derivative of the loss function with respect to f(x)
        self.output_layer.cached_activation_gradient = -1 * (
                    np.eye(self.n_labels)[train_y].T / self.output_layer.cached_activation_output_forward)
        # derivative of the loss function with respect to a(x) of the output function
        self.output_layer.cached_preactivation_gradient = -1 * (
                    np.eye(self.n_labels)[train_y].T - self.output_layer.cached_activation_output_forward)
        # updating gradient of weights and biases entering the output layer
        self.output_layer.pre_activation.dw = (1.0 / n_items_batch) * np.dot(
            self.output_layer.cached_preactivation_gradient, self.hidden_layers[-1].cached_activation_output_forward.T)
        self.output_layer.pre_activation.db = (1.0 / n_items_batch) * np.sum(
            self.output_layer.cached_preactivation_gradient, axis=1, keepdims=True)

        # gradient of the preactivation of the next layer
        parent_pre_activation_gradients = self.output_layer.cached_preactivation_gradient
        # weights of the next preactivation layer
        next_layer_weights = self.output_layer.pre_activation.weights
        for layer_indx in range(self.n_hidden - 1, -1, -1):
            if layer_indx != 0:
                prev_activation_forward = self.hidden_layers[layer_indx - 1].cached_activation_output_forward
            else:
                prev_activation_forward = train_x.T

            # applying back propagation on the hidden layer
            self.hidden_layers[layer_indx].backward( \
                prev_activation_forward, \
                next_layer_weights, \
                parent_pre_activation_gradients)

            # setting the parent pre activation gradient for the next layer in the backpropagation
            parent_pre_activation_gradients = self.hidden_layers[layer_indx].cached_preactivation_gradient
            # setting the weights to be used by the next layer's backpropagation
            next_layer_weights = self.hidden_layers[layer_indx].pre_activation.weights

    def update(self):
        # based on the object holding the gradients we will update the parameters
        for i in range(0, self.n_hidden):
            self.hidden_layers[i].pre_activation.weights = \
                self.hidden_layers[i].pre_activation.weights - self.learning_rate * self.hidden_layers[
                    i].pre_activation.dw
            self.hidden_layers[i].pre_activation.bias = \
                self.hidden_layers[i].pre_activation.bias - self.learning_rate * self.hidden_layers[i].pre_activation.db
        # updating output parameters
        self.output_layer.pre_activation.weights = \
            self.output_layer.pre_activation.weights - self.learning_rate * self.output_layer.pre_activation.dw
        self.output_layer.pre_activation.bias = \
            self.output_layer.pre_activation.bias - self.learning_rate * self.output_layer.pre_activation.db

    def train(self, verbose=1):
        # adding seed to avoid any changes in multiple runs due to different initialization
        np.random.seed(self.seed)
        self.initialize_weights(verbose)
        train_losses = []
        val_losses = []
        if verbose > 0:
            progress_bar = tqdm(total=self.num_iterations)
        for epoch in range(self.num_iterations):
            for batch_indx in range(0, self.train_x.shape[0], self.min_batch_size):
                lower_range = batch_indx
                upper_range = batch_indx + self.min_batch_size
                if upper_range > self.train_x.shape[0]:
                    upper_range = self.train_x.shape[0]
                train_x = self.train_x[lower_range: upper_range, :]
                train_y = self.train_y[lower_range: upper_range]
                self.forward(train_x)

                # computing gradients through backward propagation
                self.backward(train_x, train_y)

                # updating parameters based on gradients
                self.update()

            # computing training loss at the end of each epoch
            train_predicted_probs = self.forward(self.train_x)
            train_loss = self.loss(self.train_y, train_predicted_probs.T)
            train_losses.append(train_loss)

            # computing validation loss
            val_predicted_probs = self.forward(self.val_x)
            val_loss = self.loss(self.val_y, val_predicted_probs.T)
            val_losses.append(val_loss)
            # display train and validaton loss for each epoch
            if verbose > 0:
                progress_bar.set_postfix(
                    {'epoch': str(epoch), 'Train_loss': str(train_loss), 'val_loss': str(val_loss)})

            # Shuffling the data every epoch
            train_indices = np.random.permutation(self.train_x.shape[0])
            self.train_x = self.train_x[train_indices, :]
            self.train_y = self.train_y[train_indices]

            # showing a step in progress bar
            if verbose > 0:
                progress_bar.update()
        val_predicted_probs = self.forward(self.val_x)
        val_acc = self.accuracy(self.val_y, val_predicted_probs.T)  # accuracy on last epoch

        test_predicted_probs = self.forward(self.test_x)
        test_acc = self.accuracy(self.test_y, test_predicted_probs.T)  # accuracy on last epoch

        # plotting validation and training history
        epoch_count = range(1, self.num_iterations + 1)
        if verbose > 0:
            plt.plot(epoch_count, train_losses, 'r--')
            plt.plot(epoch_count, val_losses, 'b-')
            plt.legend(['Training Loss', 'Validation Loss'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()
        if self.model_path:
            pickle.dump(self, open(self.model_path, 'wb'))
        return (val_acc, val_loss, test_acc, test_predicted_probs.T)

    def test(self, hidden_layer_index=1):
        # used to test gradients
        random_train_indice = np.random.randint(low=0, high=self.train_x.shape[0])
        training_sample = self.train_x[random_train_indice]
        sample_label = self.train_y[random_train_indice]

        p = np.min([10, self.hidden_layers[hidden_layer_index].pre_activation.dimensions[0]])
        cached_weights = self.hidden_layers[hidden_layer_index].pre_activation.weights
        chosen_n = []
        differences = []
        for n_iter_test in range(15):
            k = np.random.randint(low=1, high=5)
            i = np.random.randint(low=0, high=5)
            N = k * (10 ** i)
            chosen_n.append(N)
            epsilon = 1.0 / N
            gradient_finite_difference = np.zeros(
                (p, self.hidden_layers[hidden_layer_index].pre_activation.dimensions[1]))
            difference = np.zeros((p, self.hidden_layers[hidden_layer_index].pre_activation.dimensions[1]))
            # we are trying on P elements each element of p represent a row in the parameters
            # testing on p rows of the weight and gradients
            for i in range(p):
                # using 3 random columns to compute their maximum difference with finite gradient
                for j in np.random.randint(0,
                                           high=self.hidden_layers[hidden_layer_index].pre_activation.dimensions[1] - 1,
                                           size=3):
                    weight_addition_matrix = np.zeros(self.hidden_layers[hidden_layer_index].dimension)
                    weight_addition_matrix[i, j] = 1
                    self.hidden_layers[
                        hidden_layer_index].pre_activation.weights = cached_weights + epsilon * weight_addition_matrix
                    # computing predicted labels with the updated weights
                    output_label = self.forward(training_sample)
                    loss_1 = self.loss(sample_label, output_label.T)

                    self.hidden_layers[
                        hidden_layer_index].pre_activation.weights = cached_weights - epsilon * weight_addition_matrix
                    output_label = self.forward(training_sample)
                    loss_2 = self.loss(sample_label, output_label.T)
                    gradient_finite_difference[i, j] = (loss_1 - loss_2) * 1.0 / 2 * epsilon
                    difference[i, j] = np.abs(
                        model.hidden_layers[hidden_layer_index].pre_activation.dw[i, j] - gradient_finite_difference[
                            i, j])
            difference = np.amax(difference, axis=(0, 1))
            differences.append(difference)
        plt.scatter(chosen_n, differences)
        plt.xlabel('N')
        plt.ylabel('Gradient Difference')
        plt.show()
        return chosen_n, differences


# training model with zeros initalization

model = NN(initialization_type='zeros')
val_acc, val_loss, test_acc, predicted_test = model.train()
