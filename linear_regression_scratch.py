from open_data import inputs_array, targets_array
import torch

class LinearRegressionScratch:
    """
        This class attempts to create a linear regression
        from scratch
    """
    def __init__(self, inputs, targets, weights, bias):
        self.inputs = inputs
        self.targets = targets
        self.weights = weights
        self.bias = bias

    def model(self):
        # @ is matrix multiplication by pytorch
        return inputs @ weights.t() + bias

    def MSE(self):
        difference = self.model() - targets
        difference_square = difference*difference
        return torch.sum(difference_square/difference.numel())


if __name__ == "__main__":
    inputs = torch.from_numpy(inputs_array)
    targets = torch.from_numpy(targets_array)

    # We need 506 randoms weights for each row and column
    weights = torch.randn(506, 13, requires_grad=True)

    # We need 506 biases added to each bias
    bias = torch.randn(506, requires_grad=True)
    linear = LinearRegressionScratch(inputs, targets, weights, bias)
    print(linear.targets)
    print(linear.model())

    print(linear.MSE())




