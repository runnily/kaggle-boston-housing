from open_data import inputs_array, targets_array
import torch
import numpy
import pandas as pd

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
        self.loss = 0
        self.TOL = 1e-4

    def model(self):
        # @ is matrix multiplication by pytorch
        return inputs @ self.weights.t() + self.bias

    def MSE(self):
        # self.model () returns the predictions
        difference = self.model() - self.targets
        difference_square = difference*difference
        return torch.sum(difference_square) / difference.numel()
    
    def gradient(self):
         with torch.no_grad():
            # Update weights by subtracting a small quantity proportional to the gradient 
            self.weights -= self.weights.grad * 1e-6
            # Update bias by subtracting a small quantity proportional to the gradient
            self.bias -= self.bias.grad * 1e-6
            # resets gradients back to 0
            self.weights.grad.zero_()
            self.bias.grad.zero_()
    
    def epochs(self):
        notEqual = True
        while notEqual:
            lossTemp = self.loss
            self.loss = self.MSE() #caculate loss
            # computes the gradient for every parameter which requires_grad is set to true
            self.loss.backward()   
            print(self.loss)
            self.gradient()
            if (abs(self.loss-lossTemp)<=self.TOL):
                notEqual = False
        

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
    linear.epochs()

    preds = linear.model()
    preds_array = preds.detach().numpy()

    df = pd.DataFrame(data=preds_array, columns=["medv"])
    df.to_csv('Data/preds/boston_preds.csv')







