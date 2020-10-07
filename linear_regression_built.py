from open_data import inputs_array, targets_array
import torch

class LinearRegressionBuilt:
    """
        This class explores linear regression built in.
    """

    def __init__(self, input, targets, weights, bias):
        self.input = input
        self.targets = targets
        self.weights = weights 
        self.bias = bias
    
    
