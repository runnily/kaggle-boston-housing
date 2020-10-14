from open_data import inputs_array, targets_array
from torch import from_numpy, randn, optim
import torch.nn as nn
from torch.nn.functional import mse_loss as mse
import pandas as pd
# This allows us to get our data as tuples.
from torch.utils.data import TensorDataset, DataLoader

class LinearRegressionBuilt:
    """
        This class explores linear regression built in.
    """

    def __init__(self, inputs, targets, weights, bias):
        self.inputs = inputs
        self.targets = targets
        self.weights = weights 
        self.bias = bias

        self.train_ds = TensorDataset(self.inputs, self.targets) 
        # After using the tensor dataset which split our inputs and 
        # Each corresponding input is put in a pair with the targets
        # There for we should have 506 tuples.
        # targets into input we use the dataloader to split
        # our tuples into batches
        self.train_b = DataLoader(self.train_ds, 11, shuffle=True)

    def batch(self):
        for input, target in self.train_b:
            print(input)
            print(target)
            break

    def model(self):
        # Instead of creating our model we can use nn.Linear
        # nn.Linear(number of inputs, number of outputs)
        # Inside the model it automatically creates weights
        # Instead the weight it automatically creates bias
        # We would have 13 weights and 1 bias.
        # y= xA^T + b; x=input A=weights, b=bias
        return nn.Linear(13,1)
        
    def parameters(self):
        model = self.model()
        print(model.weight)
        print(model.bias)
        print(model.parameters())

    
    def MSE(self, preds):
        # We are using a build in loss function
        return mse(preds, self.targets)

    def train(self):
        # tell us which matrix need to be updated
        model = self.model()
        opt= optim.SGD(model.parameters(), lr=1e-6)
        for epoch in range(500):
            for inputs, outputs in self.train_b:
                # Get predictions from model
                preds = model(self.inputs)
                # Calc the loss
                loss = self.MSE(preds)
                # Calc the gradient
                loss.backward()
                # Update parameters
                opt.step()
                # Reset the gradient back to 0
                opt.zero_grad()
            print(loss.item())
        return model(self.inputs)
    


if __name__ == "__main__":
    inputs = from_numpy(inputs_array)
    targets = from_numpy(targets_array)

    weights = randn(1,13, requires_grad=True)
    bias = randn(1, requires_grad=True)
    
    r = LinearRegressionBuilt(inputs,targets,weights,bias)
    #r.batch()
    #r.parameters()
   
    preds = r.train()
    preds_array = preds.detach().numpy()

    df = pd.DataFrame(data=preds_array, columns=["medv"])
    df.to_csv('Data/preds_2/boston_preds.csv')

