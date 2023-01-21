import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from MyNeuralNetwork import *


# h = torch.load("h.pt")
h = NeuralNetwork(h_bar=h_bar).to(device)


#################### training function ############################

def train_network(model, optimizer, criterion, num_epochs):
    
    train_accuracy=[]
    test_accuracy=[]

    for epoch in range(num_epochs):

        sample_count = 0
        train_loss=[]
        for i in np.arange(-2, 2, 0.1):
            for j in np.arange(-1,1,0.1):
                
                sample_count += 1
                X_train = torch.tensor([[i, j]], dtype=torch.float, requires_grad=True, device=device)
        
                #forward feed
                # output_train = model(X_train)

                dhdx = get_dhdx(X_train, model)
                C_bar_s = C_bar(X_train, model, u_v1, u_v2, dhdx)
                h_bar_s = h_bar(X_train)


                # train_accuracy.append(torch.norm(output_train - y_train))

                #calculate the loss
                loss = criterion(h_bar_s, C_bar_s)

                train_loss.append(loss.item())

                

                #backward propagation: calculate gradients
                if loss > 0:
                    #clear out the gradients from the last step loss.backward()
                    optimizer.zero_grad()

                    loss.backward()

                    #update the weights
                    optimizer.step()

        # with torch.no_grad():
        #     output_test = model(X_test)
        #     test_loss = torch.norm(output_test - y_test)
        train_loss = np.array(train_loss)
        violate_points_num = sum(train_loss > 0)
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Average train Loss: {violate_points_num}")

    return train_loss, train_accuracy, test_accuracy


######################### start to train ##########################

optimizer = torch.optim.Adam(h.parameters(), lr=0.0001, weight_decay=1e-4)

train_loss, train_accuracy, test_accuracy = train_network(model=h, optimizer=optimizer, criterion=myCustomLoss, num_epochs=10)

torch.save(h, "h.pt")


N_samples = 5
X_train = torch.rand((N_samples,2), dtype=torch.float, device=device, requires_grad=True)
X_train = X_train * torch.tensor([4, 2], dtype=torch.float).reshape((1,2)).to(device) - torch.tensor([2,1], dtype=torch.float).reshape((1,2)).to(device)

y_train = h(X_train)
print(X_train)
print(y_train)
