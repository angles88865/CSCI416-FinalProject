import torch
import torchvision
import torchvision.transforms as transforms
import PIL
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing


def model():

    # Define a neural network class called 'Net' which inherits from 'nn.Module', the base class for all neural network modules.
    class Net(nn.Module):
        # The constructor for the 'Net' class. This is where we define all the layers and operations.
        def __init__(self):
            super(Net, self).__init__()  # Initialize the superclass.

            # Define convolutional layers with specified number of input channels, output channels, kernel size, and padding.
            self.conv1 = nn.Conv2d(3, 128, 5,
                                   padding=2)  # First convolutional layer with input channels 3 (RGB image), 128 output channels, 5x5 kernel, and padding of 2.
            self.conv2 = nn.Conv2d(128, 128, 5,
                                   padding=2)  # Second convolutional layer with 128 channels in and out, same kernel size and padding.
            self.conv3 = nn.Conv2d(128, 256, 3,
                                   padding=1)  # Third convolutional layer increasing channels to 256 with a 3x3 kernel.
            self.conv4 = nn.Conv2d(256, 256, 3,
                                   padding=1)  # Fourth convolutional layer, same channel size with a 3x3 kernel.

            # Define a pooling layer to reduce the spatial dimensions of the output from the convolutional layers.
            self.pool = nn.MaxPool2d(2,
                                     2)  # Max pooling with a 2x2 window and stride of 2, effectively reducing the spatial dimensions by half.

            # Define batch normalization layers for convolutional and dense layers to stabilize and speed up training.
            self.bn_conv1 = nn.BatchNorm2d(128)
            self.bn_conv2 = nn.BatchNorm2d(128)
            self.bn_conv3 = nn.BatchNorm2d(256)
            self.bn_conv4 = nn.BatchNorm2d(256)
            self.bn_dense1 = nn.BatchNorm1d(1024)
            self.bn_dense2 = nn.BatchNorm1d(512)

            # Define dropout layers to prevent overfitting by randomly zeroing some of the elements of the input tensor during training using samples from a Bernoulli distribution.
            self.dropout_conv = nn.Dropout2d(p=0.25)  # Dropout for convolutional layers.
            self.dropout = nn.Dropout(p=0.5)  # Dropout for dense (fully connected) layers.


            self.fc1 = nn.Linear(256 * 50 * 50,
                                 1024)  # Flatten the output from the conv layers and connect to 1024 neurons.
            self.fc2 = nn.Linear(1024, 512)  # Connect the 1024 neurons to 512 neurons.
            self.fc3 = nn.Linear(512,
                                 10)  # Finally, connect to 10 neurons corresponding to the number of classes in the dataset.

        # Define the method for the forward pass through the convolutional layers.
        def conv_layers(self, x):
            out = F.relu(self.bn_conv1(
                self.conv1(x)))  # Apply the first convolutional layer, then batch normalization, then ReLU activation.
            out = F.relu(self.bn_conv2(self.conv2(out)))  # Same for second convolutional layer.
            out = self.pool(out)  # Apply max pooling to reduce spatial dimensions.
            out = self.dropout_conv(out)  # Apply dropout to the features.

            # Repeat the process for the third and fourth convolutional layers.
            out = F.relu(self.bn_conv3(self.conv3(out)))
            out = F.relu(self.bn_conv4(self.conv4(out)))
            out = self.pool(out)
            out = self.dropout_conv(out)
            return out

        # Define the method for the forward pass through the dense layers.
        def dense_layers(self, x):
            print(x.shape)
            x = x.view(x.size(0), -1)  # Flatten the tensor
            out = F.relu(self.bn_dense1(
                self.fc1(x)))  # Apply the first fully connected layer, batch normalization, then ReLU activation.
            out = self.dropout(out)  # Apply dropout.

            # Apply the second fully connected layer, batch normalization, then ReLU activation.
            out = F.relu(self.bn_dense2(self.fc2(out)))
            out = self.dropout(out)  # Apply dropout.

            out = self.fc3(out)  # Apply the final fully connected layer to get the output.
            return out

        # Define the forward method that specifies how the input tensor flows through the network.
        def forward(self, x):
            conv_out = self.conv_layers(x)

            # Flatten the tensor
            flattened = conv_out.view(conv_out.size(0), -1)

            out = self.dense_layers(
                self.conv_layers(x))  # Pass the input through the convolutional layers and the dense layers.
            return out

    # Instantiate the Net class and move it to the device
    # Assuming you are using a GPU, otherwise use 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    # Define the loss function and the optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                           amsgrad=False)
    # Learning rate scheduler which reduces the learning rate when a metric has stopped improving.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=0)

    # Initialize lists to keep track of loss and accuracy for training and validation.
    loss_hist, acc_hist = [], []
    loss_hist_val, acc_hist_val = [], []

    print('start training ...')

    # Start the training loop.
    for epoch in range(40):
        running_loss = 0.0
        correct = 0
        # Loop over the data iterator, and feed the inputs to the network and optimize.
        for data in train_loader:
            batch, labels = data
            batch, labels = batch.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients.
            outputs = net(batch)  # Get the output from the network.
            loss = criterion(outputs, labels)  # Calculate the loss.
            loss.backward()  # Backpropagate the error.
            optimizer.step()  # Optimize the network.

            # Compute training statistics.
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        # Calculate average loss and accuracy over the training dataset.
        avg_loss = running_loss / len(train_set)
        avg_acc = correct / len(train_set)
        loss_hist.append(avg_loss)
        acc_hist.append(avg_acc)

        # Switch to evaluation mode for validation statistics.
        net.eval()
        with torch.no_grad():  # Disable gradient calculation.
            loss_val = 0.0
            correct_val = 0
            # Repeat the above process for the validation data.
            for data in val_loader:
                batch, labels = data
                batch, labels = batch.to(device), labels.to(device)
                outputs = net(batch)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                loss_val += loss.item()

            # Calculate average loss and accuracy over the validation dataset.
            avg_loss_val = loss_val / len(val_set)
            avg_acc_val = correct_val / len(val_set)
            loss_hist_val.append(avg_loss_val)
            acc_hist_val.append(avg_acc_val)

        # Switch back to training mode.
        net.train()

        # Step the scheduler.
        scheduler.step(avg_loss_val)

        # Print statistics after each epoch.
        print('[epoch %d] loss: %.5f accuracy: %.4f val loss: %.5f val accuracy: %.4f' %
              (epoch + 1, avg_loss, avg_acc, avg_loss_val, avg_acc_val))


if __name__ == '__main__':
    model()
    print('done')
