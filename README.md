#   NdLinear Neural Net

A neural net composed of ndlinear layers

## Description

Implements a deep network (Multi-layer Perceptron) capable of classifying handwritten digits using the pytorch and ndlinear libraries, with an accuracy of 95% on Mnist dataset.

### Dependencies

* torch
* torchvision
* ndlinear
* matplotlib.pyplot

### Executing program

* Install ndlinear package
```
!pip install ndlinear
```
* Import dependecies check for GPU
```
%matplotlib inline

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from ndlinear import NdLinear
import matplotlib.pyplot as plt

print('Using PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print('No GPU found, using CPU instead.')
    device = torch.device('cpu')
```
* Load Mnist data set and create train/test loaders
```
batch_size = 10
data_dir = './data'
train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=ToTensor())
test_dataset = datasets.MNIST(data_dir, train=False, transform=ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
```
* Exploratory Data Analysis reveals data is a 4th order float tensor, label is a 1st order long Tensor
```
for (Input, Label) in train_loader:
  print('data:', Input.size(), 'type', Input.type())
  print('label:', Label.size(), 'type', Label.type())
  break

#visualize data
pltsize = 1
plt.figure(figsize=(10*pltsize, pltsize))

for i in range(10):
  plt.subplot(1,10,i+1)
  plt.axis('off')
  plt.imshow(Input[i,:,:,:].numpy().reshape(28,28), cmap ='gray_r')
  plt.title('Class: '+str(Label[i].item()))
```
* The model definition specifies a 2 layer Multilayer Perceptron utilizing a ReLu activation function
```
class NdLinearMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            NdLinear(input_dims=(784,), hidden_size=(20,)),
            nn.ReLU(),
            NdLinear(input_dims=(20,), hidden_size=(10,)),
        )

    def forward(self, x):
        return self.layers(x)
```
* Instanstiate model and define hyperparams
```
mlp = NdLinearMLP().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
```
* Define function to count correct predictions
```
def correct(output, label):
  predicted_labels = output.argmax(1)
  correct_ones = (predicted_labels == label).type(torch.float)
  return correct_ones.sum().item()
```
* Define model training function
```
def train(data_loader, model, criterion, optimizer):
  model.train()

  num_batches = len(data_loader)
  num_items = len(data_loader.dataset)

  total_loss = 0
  total_correct = 0
  for input, label in data_loader:
    #Copy to GPU
    input = input.to(device)
    label = label.to(device)
    #forward pass
    output = model(input)

    #calculate the loss
    loss = criterion(output, label)
    total_loss += loss

    #count number of correct
    total_correct += correct(output, label)

    #backprop
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  train_loss = total_loss/num_batches
  accuracy = total_correct/num_items
  print(f'Average loss: {train_loss:7f}, accuracy: {accuracy:.2%}')
```
* Model Training loop
```
%%time

epochs = 10
for epoch in range(epochs):
  print(f'Training epoch: {epoch+1}')
  train(train_loader, mlp, loss_fn, optimizer)
```
* Define function to evaluate the model(basically same as training without backpropagation) and tally correct predictions in test set
```
def evaluate(test_loader, model, criterion):
  mlp.eval()

  num_batches = len(test_loader)
  num_items = len(test_loader.dataset)

  test_loss = 0
  total_correct = 0

  with torch.no_grad():
    for input, label in test_loader:
      #Copy to GPU
      input = input.to(device)
      label = label.to(device)

      #forward pass
      output = model(input)

      #calculate loss
      loss = criterion(output, label)
      test_loss += loss.item()

      #Count correct digits
      total_correct += correct(output, label)

  test_loss = test_loss/num_batches
  accuracy = total_correct/num_items

  print(f"Testset accuracy: {100*accuracy:>0.1f}%, average loss: {test_loss:>7f}")
```
* Define function to make predictions against single items in test set and visualize results
```
def test_predict(data_loader, model):
  for (input, label) in train_loader:
    break

  pltsize = 1
  plt.figure(figsize=(10*pltsize, pltsize))

  for i in range(1):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    output = model(input[i,:,:,:])
    plt.imshow(input[i,:,:,:].numpy().reshape(28,28), cmap ='gray_r')
    predict = output.argmax(1).tolist()
    predict = predict[0]
    plt.title(f'Actual: {str(label[i].item())}, Prediction: {str(predict)}')
```
* Evaluate the model against unseen data by passing the test_loader
```
evaluate(test_loader, mlp, loss_fn)
```
* Visualize model inference 
```
test_predict(test_loader, mlp)
```

## Author

Samuel Kalu
  
* email : [samkalu@ttu.edu](mailto:samkalu@ttu.edu)
* linkedin : [@SamuelKalu](https://www.linkedin.com/in/samuel-kalu-74a359342/)


## Acknowledgments

Inspiration, code snippets, etc.
* [Samson Zhang](https://www.youtube.com/watch?v=w8yWXqWQYmU&t=187s&ab_channel=SamsonZhang)
* [Ensemble Ai](https://github.com/ensemble-core)
* [CSC - IT Center for Science](https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/pytorch-mnist-mlp.ipynb)

