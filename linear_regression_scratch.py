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
    
    def gradient(self):
        with torch.no_grad():
            # subtracting a small quantity proportional to the gradient 
            self.weights -= self.weights.grad * 1e-5
            #  subtracting a small quantity proportional to the gradient
            self.bias -= self.bias.grad * 1e-5
            # resets gradients back to 0
            weights.grad.zero_()
            bias.grad.zero_()
    
    def epochs(self):
        loss = self.MSE()
        # loss computes the gradient for every parameter which requires_grad is set to true
        loss.backward()
        for _ in range(300):
            self.gradient()
        return loss


if __name__ == "__main__":
    inputs = torch.from_numpy(inputs_array)
    targets = torch.from_numpy(targets_array)

    # There is 1 target value so we need 1 row of random weights
    # for each input which is multipled
    weights = torch.randn(1, 13, requires_grad=True)

    # There is 1 target value so we need 1 bias added to
    # each caculation
    bias = torch.randn(1, requires_grad=True)

    linear = LinearRegressionScratch(inputs, targets, weights, bias)

    print(linear.epochs())




