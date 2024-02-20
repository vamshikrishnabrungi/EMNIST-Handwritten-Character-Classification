

# Imported libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from sklearn.model_selection import KFold
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.optim as optim

# Define transformations to apply to each image in dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(28, 28).t().contiguous().view(1, 28, 28)), # Transpose the image back
    transforms.Normalize((0.1307,), (0.3081,))
])

# Define dataset path
data_path = './data'

# Load "EMNIST "Balanced""" dataset
train_dataset = datasets.EMNIST(
    data_path, split='balanced', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST(
    data_path, split='balanced', train=False, download=True, transform=transform)

# Print number of training and testing sets' samples
print('Number of training samples:', len(train_dataset))
print('Number of testing samples:', len(test_dataset))

# Read mapping file to map the label indices to character names
mapping_file = open('emnist-balanced-mapping.txt')
mapping = {}
for line in mapping_file:
    parts = line.split()
    mapping[int(parts[0])] = chr(int(parts[1]))

# Plot figures to visualise some of the dataset samples
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]
    img = img.squeeze().numpy()
    ax.imshow(img, cmap='gray')
    ax.set_title(mapping[label])
    ax.axis('off')
plt.show()

# we check if CUDA is available or not 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#here we implement the cnn 

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 47)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
import time
import random
import numpy as np
import torch



seed = 42  
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

k = 3  # number of folds
learning_rate = 0.001  # adaptive learning rate
activation_fn = F.relu  # ReLU activation function
optimizer_fn = torch.optim.Adam  # Adam optimizer
batch_norm = True  # we use batch normalization
dropout = None  # we do not use dropout here
regularization = 1e-5  # L1 regularization strength
num_epochs = 10  # number of epochs
batch_size = 64  # batch size

import matplotlib.pyplot as plt
import numpy as np

# Defining the lists to store the loss and accuracy values for each epoch and fold
train_losses, val_losses = [], []
train_accs, val_accs = [], []

# Creating  the KFold object
kf = KFold(n_splits=k, shuffle=True)

# Initializing a variable to store the total training loss across all folds
total_train_loss = 0.0

start_time = time.time()  
# Looping over the folds
for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f'Fold {fold+1}')
    print(f'Hyperparameters: learning_rate={learning_rate}, activation_fn={activation_fn.__name__}, optimizer_fn={optimizer_fn.__name__}, batch_norm={batch_norm}, dropout={dropout}, regularization={regularization}')
    
    # Creating the data loaders for this fold
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))

    # Initializing your CNN model
    model = CNN()  

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_fn(model.parameters(), lr=learning_rate, weight_decay=regularization)

    # Train and evaluate your model on this fold
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if batch_norm:
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        loss += regularization * torch.norm(module.weight, 1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        train_loss /= len(train_idx)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if batch_norm:
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        loss += regularization * torch.norm(module.weight, 1)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += targets.size(0)
            val_correct += (predicted == targets).sum().item()
        val_loss /= len(val_idx)
        val_acc = val_correct / val_total

        # Adding the training loss for this fold to the total training loss
        total_train_loss += train_loss

        # Appending the loss and accuracy values to the lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Printing the training and validation loss and accuracy for this epoch
        print(f'Fold {fold+1}, Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')

        # Save the model after each fold
        torch.save(model.state_dict(), "model.pt")

# Calculating the average training loss across all folds
avg_train_loss = total_train_loss / k
# Print the average training loss
print(f'Average Training Loss: {avg_train_loss:.4f}')
end_time = time.time()  # End time for training
print(f"Training time: {end_time - start_time:.2f} seconds")

# Ploting the loss function graph
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Ploting the accuracy graph
plt.plot(train_accs, label='Training Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from sklearn.model_selection import KFold
import torch.nn.functional as F
import torch
from torch.utils.data import SubsetRandomSampler
import random
import numpy as np



k = 3  # number of folds
learning_rate = 0.01  # adaptive learning rate
activation_fn = F.leaky_relu  # LeakyReLU activation function
optimizer_fn = torch.optim.SGD  # Stochastic gradient descent optimizer
batch_norm = True  # now we use batch normalization
dropout = 0.2  # use dropout with 20% probability of dropping out a neuron
regularization = 1e-4  # L2 regularization strength
num_epochs = 10  # number of epochs
batch_size = 64  # batch size

import matplotlib.pyplot as plt
import numpy as np

# Defining the lists to store the loss and accuracy values for each epoch and fold
train_losses, val_losses = [], []
train_accs, val_accs = [], []

import time

# Creating the KFold object
kf = KFold(n_splits=k, shuffle=True)

# Initializing a variable to store the total training loss across all folds
total_train_loss = 0.0
start_time = time.time()  
# Looping over the folds
for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f'Fold {fold+1}')
    print(f'Hyperparameters: learning_rate={learning_rate}, activation_fn={activation_fn.__name__}, optimizer_fn={optimizer_fn.__name__}, batch_norm={batch_norm}, dropout={dropout}, regularization={regularization}')
    
    # Creating the data loaders for this fold
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))

    # Initializing your CNN model
    model = CNN()  # replace MyCNN with your CNN model class

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_fn(model.parameters(), lr=learning_rate, weight_decay=regularization)



    # Training and evaluating your model on this fold
    start_time = time.time()  # record the start time

    # Training and evaluate your model on this fold
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if batch_norm:
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        loss += regularization * torch.norm(module.weight, 1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        train_loss /= len(train_idx)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if batch_norm:
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        loss += regularization * torch.norm(module.weight, 1)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += targets.size(0)
            val_correct += (predicted == targets).sum().item()
        val_loss /= len(val_idx)
        val_acc = val_correct / val_total
        
        # Adding the training loss for this fold to the total training loss
        total_train_loss += train_loss

        # Appending the loss and accuracy values to the lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)



        # Printing the training and validation loss and accuracy for this epoch
        print(f'Fold {fold+1}, Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')


        # Printing the training time for this fold
        end_time = time.time()
        training_time = end_time - start_time
        print(f'Fold {fold+1} training time: {training_time:.2f} seconds')
        print('')

        # Saving the model after each fold
        torch.save(model.state_dict(), f'model_fold{fold+1}.pt')



# Calculating the average training loss across all folds
avg_train_loss = total_train_loss / k

end_time = time.time()  # End time for training
print(f"Training time: {end_time - start_time:.2f} seconds")

# Ploting the loss function graph
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Ploting the accuracy graph
plt.plot(train_accs, label='Training Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from sklearn.model_selection import KFold
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler
import time
k = 3  # number of folds
learning_rate = 0.1  # adaptive learning rate
activation_fn = F.elu  # ELU activation function
optimizer_fn = torch.optim.RMSprop  # RMSprop optimizer
batch_norm = False  # do not use batch normalization
dropout = 0.0  # do not use dropout
regularization = 0.0  # no L2 regularization
num_epochs = 10  # number of epochs
batch_size = 64  # batch size

# Defining the lists to store the loss and accuracy values for each epoch and fold
train_losses, val_losses = [], []
train_accs, val_accs = [], []

# Creating the KFold object
kf = KFold(n_splits=k, shuffle=True)
# Initializing a variable to store the total training loss across all folds
total_train_loss = 0.0
start_time = time.time()  
# Looping over the folds
for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f'Fold {fold+1}')
    print(f'Hyperparameters: learning_rate={learning_rate}, activation_fn={activation_fn.__name__}, optimizer_fn={optimizer_fn.__name__}, batch_norm={batch_norm}, dropout={dropout}, regularization={regularization}')
    
    # Creating the data loaders for this fold
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))

    # Initializing your CNN model
    model = CNN()  # replace MyCNN with your CNN model class

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_fn(model.parameters(), lr=learning_rate, weight_decay=regularization)

    # Training and evaluate your model on this fold
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if batch_norm:
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        loss += regularization * torch.norm(module.weight, 1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        train_loss /= len(train_idx)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if batch_norm:
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        loss += regularization * torch.norm(module.weight, 1)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += targets.size(0)
            val_correct += (predicted == targets).sum().item()
        val_loss /= len(val_idx)
        val_acc = val_correct / val_total

        # Adding the training loss for this fold to the total training loss
        total_train_loss += train_loss

        # Append the loss and accuracy values to the lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Printing the training and validation loss and accuracy for this epoch
        print(f'Fold {fold+1}, Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')


# Calculating the average training loss across all folds
avg_train_loss = total_train_loss / k
end_time = time.time()  # End time for training
print(f"Training time: {end_time - start_time:.2f} seconds")
# Ploting the loss function graph
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Ploting the accuracy graph
plt.plot(train_accs, label='Training Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



# Loading the saved model state dictionary
saved_state_dict = torch.load('model.pt')

# Creating a new instance of your CNN model
model = CNN()

# Loading the saved state dictionary into the model
model.load_state_dict(saved_state_dict)

# Seting the model to evaluation mode
model.eval()

# Looping over the first six samples in the testing dataset and make predictions
num_samples = 6
for i in range(num_samples):
    inputs, target = test_dataset[i]
    output = model(inputs.unsqueeze(0))  # add batch dimension
    _, predicted = torch.max(output.data, 1)
    print(f'True label: {target}, Predicted label: {predicted.item()}')


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 47)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.bn1 = nn.BatchNorm1d(128) if batch_norm else None
        self.bn2 = nn.BatchNorm1d(64) if batch_norm else None
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        if batch_norm:
            x = self.bn1(x)
        x = activation_fn(x)
        if dropout is not None:
            x = self.dropout(x)
        x = self.fc2(x)
        if batch_norm:
            x = self.bn2(x)
        x = activation_fn(x)
        if dropout is not None:
            x = self.dropout(x)
        x = self.fc3(x)
        return x



from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
import time
import random
import numpy as np
import torch



seed = 42  
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

k = 3  # number of folds
learning_rate = 0.001  # adaptive learning rate
activation_fn = F.relu  # ReLU activation function
optimizer_fn = torch.optim.Adam  # Adam optimizer
batch_norm = True  # use batch normalization
dropout = None  # do not use dropout
regularization = 1e-5  # L1 regularization strength
num_epochs = 10  # number of epochs
batch_size = 64  # batch size

import matplotlib.pyplot as plt
import numpy as np

# Defining the lists to store the loss and accuracy values for each epoch and fold
train_losses, val_losses = [], []
train_accs, val_accs = [], []

# Create the KFold object
kf = KFold(n_splits=k, shuffle=True)

# Initialize a variable to store the total training loss across all folds
total_train_loss = 0.0

start_time = time.time()  # Start time for training
# Looping over the folds
for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f'Fold {fold+1}')
    print(f'Hyperparameters: learning_rate={learning_rate}, activation_fn={activation_fn.__name__}, optimizer_fn={optimizer_fn.__name__}, batch_norm={batch_norm}, dropout={dropout}, regularization={regularization}')
    
    # Creating the data loaders for this fold
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))

    # Initializing your CNN model
    model = MLP()  # replace MyCNN with your CNN model class

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_fn(model.parameters(), lr=learning_rate, weight_decay=regularization)

    # Training and evaluate your model on this fold
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if batch_norm:
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        loss += regularization * torch.norm(module.weight, 1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        train_loss /= len(train_idx)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if batch_norm:
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        loss += regularization * torch.norm(module.weight, 1)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += targets.size(0)
            val_correct += (predicted == targets).sum().item()
        val_loss /= len(val_idx)
        val_acc = val_correct / val_total

        # Adding the training loss for this fold to the total training loss
        total_train_loss += train_loss

        # Appending the loss and accuracy values to the lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Printing the training and validation loss and accuracy for this epoch
        print(f'Fold {fold+1}, Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')

        # Saving the model after each fold
        torch.save(model.state_dict(), "2model.pt")

# Calculating the average training loss across all folds
avg_train_loss = total_train_loss / k
# Print the average training loss
print(f'Average Training Loss: {avg_train_loss:.4f}')
end_time = time.time()  # End time for training
print(f"Training time: {end_time - start_time:.2f} seconds")

# Ploting the loss function graph
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Ploting the accuracy graph
plt.plot(train_accs, label='Training Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from sklearn.model_selection import KFold
import torch.nn.functional as F
import torch
from torch.utils.data import SubsetRandomSampler
import random
import numpy as np



k = 3  # number of folds
learning_rate = 0.01  # adaptive learning rate
activation_fn = F.leaky_relu  # LeakyReLU activation function
optimizer_fn = torch.optim.SGD  # Stochastic gradient descent optimizer
batch_norm = True  # use batch normalization
dropout = 0.2  # use dropout with 20% probability of dropping out a neuron
regularization = 1e-4  # L2 regularization strength
num_epochs = 10  # number of epochs
batch_size = 64  # batch size

import matplotlib.pyplot as plt
import numpy as np

# Defining the lists to store the loss and accuracy values for each epoch and fold
train_losses, val_losses = [], []
train_accs, val_accs = [], []

import time

# Creating the KFold object
kf = KFold(n_splits=k, shuffle=True)

# Initializing a variable to store the total training loss across all folds
total_train_loss = 0.0
start_time = time.time()  # Start time for training
# Looping over the folds
for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f'Fold {fold+1}')
    print(f'Hyperparameters: learning_rate={learning_rate}, activation_fn={activation_fn.__name__}, optimizer_fn={optimizer_fn.__name__}, batch_norm={batch_norm}, dropout={dropout}, regularization={regularization}')
    
    # Creating the data loaders for this fold
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))

    # Initializing your CNN model
    model = MLP()  # replace MyCNN with your CNN model class

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_fn(model.parameters(), lr=learning_rate, weight_decay=regularization)

    # Training and evaluate your model on this fold
    start_time = time.time()  # record the start time

    # Training and evaluate your model on this fold
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if batch_norm:
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        loss += regularization * torch.norm(module.weight, 1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        train_loss /= len(train_idx)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if batch_norm:
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        loss += regularization * torch.norm(module.weight, 1)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += targets.size(0)
            val_correct += (predicted == targets).sum().item()
        val_loss /= len(val_idx)
        val_acc = val_correct / val_total
        
        # Adding the training loss for this fold to the total training loss
        total_train_loss += train_loss

        # Appending the loss and accuracy values to the lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)



        # Printing the training and validation loss and accuracy for this epoch
        print(f'Fold {fold+1}, Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')


        # Printing the training time for this fold
        end_time = time.time()
        training_time = end_time - start_time
        print(f'Fold {fold+1} training time: {training_time:.2f} seconds')
        print('')

        # Saving the model after each fold
        torch.save(model.state_dict(), f'model_fold{fold+1}.pt')



# Calculating the average training loss across all folds
avg_train_loss = total_train_loss / k

end_time = time.time()  # End time for training
print(f"Training time: {end_time - start_time:.2f} seconds")

# Ploting the loss function graph
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Ploting the accuracy graph
plt.plot(train_accs, label='Training Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from sklearn.model_selection import KFold
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler
import time
k = 3  # number of folds
learning_rate = 0.1  # adaptive learning rate
activation_fn = F.elu  # ELU activation function
optimizer_fn = torch.optim.RMSprop  # RMSprop optimizer
batch_norm = False  # do not use batch normalization
dropout = 0.0  # do not use dropout
regularization = 0.0  # no L2 regularization
num_epochs = 10  # number of epochs
batch_size = 64  # batch size

# Defining the lists to store the loss and accuracy values for each epoch and fold
train_losses, val_losses = [], []
train_accs, val_accs = [], []

# Creating the KFold object
kf = KFold(n_splits=k, shuffle=True)
# Initializing a variable to store the total training loss across all folds
total_train_loss = 0.0
start_time = time.time()  # Start time for training
# Looping over the folds
for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f'Fold {fold+1}')
    print(f'Hyperparameters: learning_rate={learning_rate}, activation_fn={activation_fn.__name__}, optimizer_fn={optimizer_fn.__name__}, batch_norm={batch_norm}, dropout={dropout}, regularization={regularization}')
    
    # Creating the data loaders for this fold
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))

    # Initializing your CNN model
    model = MLP()  # replace MyCNN with your CNN model class

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_fn(model.parameters(), lr=learning_rate, weight_decay=regularization)

    # Train and evaluate your model on this fold
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if batch_norm:
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        loss += regularization * torch.norm(module.weight, 1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        train_loss /= len(train_idx)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if batch_norm:
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        loss += regularization * torch.norm(module.weight, 1)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += targets.size(0)
            val_correct += (predicted == targets).sum().item()
        val_loss /= len(val_idx)
        val_acc = val_correct / val_total

        # Add the training loss for this fold to the total training loss
        total_train_loss += train_loss

        # Append the loss and accuracy values to the lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print the training and validation loss and accuracy for this epoch
        print(f'Fold {fold+1}, Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')


# Calculating the average training loss across all folds
avg_train_loss = total_train_loss / k
end_time = time.time()  # End time for training
print(f"Training time: {end_time - start_time:.2f} seconds")
# Ploting the loss function graph
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Ploting the accuracy graph
plt.plot(train_accs, label='Training Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loading the saved model state dictionary
saved_state_dict = torch.load('2model.pt')

# Creating a new instance of your CNN model
model = CNN()

# Loading the saved state dictionary into the model
model.load_state_dict(saved_state_dict)

# Seting the model to evaluation mode
model.eval()

# Looping over the first six samples in the testing dataset and make predictions
num_samples = 6
for i in range(num_samples):
    inputs, target = test_dataset[i]
    output = model(inputs.unsqueeze(0))  # add batch dimension
    _, predicted = torch.max(output.data, 1)
    print(f'True label: {target}, Predicted label: {predicted.item()}')


import torch
import torchvision
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# Loading the saved MLP and CNN models
mlp_model = torch.load("2model.pt")
cnn_model = torch.load("model.pt")

# Define the test dataset
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# Loading the test data
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

# Evaluating the MLP model
mlp_model.eval()
with torch.no_grad():
    mlp_predictions = []
    mlp_targets = []
    for images, labels in test_loader:
        outputs = mlp_model(images)
        _, predicted = torch.max(outputs.data, 1)
        mlp_predictions.extend(predicted.cpu().numpy())
        mlp_targets.extend(labels.cpu().numpy())

# Evaluating the CNN model
cnn_model.eval()
with torch.no_grad():
    cnn_predictions = []
    cnn_targets = []
    for images, labels in test_loader:
        outputs = cnn_model(images)
        _, predicted = torch.max(outputs.data, 1)
        cnn_predictions.extend(predicted.cpu().numpy())
        cnn_targets.extend(labels.cpu().numpy())

# Creating confusion matrices
mlp_cm = confusion_matrix(mlp_targets, mlp_predictions)
cnn_cm = confusion_matrix(cnn_targets, cnn_predictions)

# Ploting the MLP confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(mlp_cm, cmap=plt.cm.Blues)
plt.title("MLP Confusion Matrix")
plt.colorbar()
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Ploting the CNN confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cnn_cm, cmap=plt.cm.Blues)
plt.title("CNN Confusion Matrix")
plt.colorbar()
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
