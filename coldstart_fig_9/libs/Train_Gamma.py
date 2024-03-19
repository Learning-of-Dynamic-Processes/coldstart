import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Train_NN:
    def __init__(self, X_train, Y_train, verbose=True, hidden_layers=12, layer_dimension=64, epochs=150, batch_size=128, device = torch.device('cuda')):
        l, dim_x = X_train.shape
        l, dim_y = Y_train.shape
        self.dim_x = dim_x
        self.X_train = X_train.to(device)
        self.Y_train = Y_train.to(device)
        self.train_len = l
        self.device = device
        EPOCHS = epochs
        BATCH_SIZE = batch_size
        VERBOSE = verbose
        VALIDATION_SPLIT = 0.2

        # Defining the structure of the feedforward neural network using PyTorch.
        self.model = nn.Sequential(
            nn.Linear(dim_x, layer_dimension, bias=True),
            nn.ReLU(),
        )
        for i in range(hidden_layers):
            self.model.add_module('hidden_layer_'+str(i), nn.Linear(layer_dimension, layer_dimension, bias=True))
            self.model.add_module('relu_'+str(i), nn.ReLU())
        self.model.add_module('output_layer', nn.Linear(layer_dimension, dim_y, bias=True))
        self.model.to(device)
        if VERBOSE:
            print(self.model)

        # Training with different learning rates for the Adam Optimizer
        self.criterion = nn.MSELoss()
        self.opt = optim.Adam(self.model.parameters(), lr=0.001)
        self.train_model(EPOCHS, BATCH_SIZE, VALIDATION_SPLIT)


    def train_model(self, epochs, batch_size, val, verbose=True):
        self.model.train()
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.1, patience=50, min_lr=0.0000001)
        for epoch in range(epochs):
            running_loss = 0.0
            for i in range(0, int((1-val)*self.train_len), batch_size):
                inputs = self.X_train[i:i+batch_size]
                labels = self.Y_train[i:i+batch_size]
                self.opt.zero_grad()
                prediction = self.model(inputs)
                loss = self.criterion(prediction, labels)
                loss.backward()
                self.opt.step()
                running_loss += loss.item() * inputs.size(0)
            with torch.no_grad():
                val_inputs = self.X_train[int((1-val)*self.train_len):]
                val_labels = self.Y_train[int((1-val)*self.train_len):]
                val_outputs = self.model(val_inputs)
                val_loss = self.criterion(val_outputs, val_labels)
                reduce_lr.step(val_loss)
                
            if (epoch) % 50 == 0:
                print('Epoch [{}/{}], Loss: {}'.format(epoch+1, epochs, running_loss/self.train_len))

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)
