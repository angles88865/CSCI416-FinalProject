import torch
import PIL
from torch.utils.data import random_split, DataLoader
import argparse
import time
import os
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip
from torchvision.datasets import ImageFolder
import wandb
import torch.optim as optim
from model import CNNModel
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

## input hyper-paras
parser = argparse.ArgumentParser(description="nueral networks")
parser.add_argument("-mode", dest="mode", type=str, default='train', help="train or test")
parser.add_argument("-num_epoches", dest="num_epoches", type=int, default=40, help="num of epoches")

parser.add_argument("-fc_hidden1", dest="fc_hidden1", type=int, default=50, help="dim of hidden neurons")
parser.add_argument("-fc_hidden2", dest="fc_hidden2", type=int, default=50, help="dim of hidden neurons")
parser.add_argument("-learning_rate", dest="learning_rate", type=float, default=0.0001, help="learning rate")
parser.add_argument("-decay", dest="decay", type=float, default=0.01, help="learning rate")
parser.add_argument("-batch_size", dest="batch_size", type=int, default=100, help="batch size")
parser.add_argument("-dropout", dest="dropout", type=float, default=0.4, help="dropout prob")
parser.add_argument("-rotation", dest="rotation", type=int, default=10, help="image rotation")
parser.add_argument("-activation", dest="activation", type=str, default='relu', help="activation function")
parser.add_argument("-channel_out1", dest='channel_out1', type=int, default=64, help="number of channels")
parser.add_argument("-channel_out2", dest='channel_out2', type=int, default=64, help="number of channels")
parser.add_argument("-k_size", dest='k_size', type=int, default=5, help="size of filter")
parser.add_argument("-pooling_size", dest='pooling_size', type=int, default=2, help="size for max pooling")
parser.add_argument("-stride", dest='stride', type=int, default=1, help="stride for filter")
parser.add_argument("-max_stride", dest='max_stride', type=int, default=2, help="stride for max pooling")
parser.add_argument("-ckp_path", dest='ckp_path', type=str, default="checkpoint", help="path of checkpoint")

args = parser.parse_args()

# Define a series of transformations for the training data.
train_transform = transforms.Compose([
    transforms.Resize((100, 100)),  # Resize the images to 100x100 pixels.
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the images horizontally 50% of the time.
    transforms.RandomAffine(  # Apply random affine transformations to the images.
        degrees=(-5, 5),  # Rotate by degrees between -5 and 5.
        translate=(0.1, 0.1),  # Translate by a fraction of image width/height (10% here).
        scale=(0.9, 1.1),  # Scale images between 90% and 110%.
        # resample=PIL.Image.BILINEAR  # Use bilinear interpolation for resampling.
        interpolation=PIL.Image.BILINEAR  # Use 'interpolation' instead of 'resample'
    ),
    transforms.ToTensor(),  # Convert images to PyTorch tensors.
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize tensors with mean and standard deviation.
])

# Define transformations for the test data.
test_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors.
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize tensors with mean and standard deviation.
])


def load_data(data_dir, batch_size, train_val_split=0.8):
    """
    Loads the ASL alphabet dataset and returns PyTorch DataLoaders for training, validation, and test.

    Args:
        data_dir (str): Path to the root directory of the ASL alphabet dataset.
        batch_size (int): Batch size for the data loaders.
        train_val_split (float): Proportion of the dataset to use for training (the rest for validation).

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Define the data transformations
    train_transform = Compose([
        Resize((64, 64)),
        RandomHorizontalFlip(p=0.5),  # Augmentation for training
        ToTensor()
    ])
    test_transform = Compose([
        Resize((64, 64)),
        ToTensor()
    ])

    # Paths for train and test data
    train_path = os.path.join(data_dir, 'asl_alphabet_train', 'asl_alphabet_train')
    test_path = os.path.join(data_dir, 'asl_alphabet_test', 'asl_alphabet_test')

    # Load the training dataset
    train_dataset = ImageFolder(root=train_path, transform=train_transform)

    # Split the training dataset into training and validation sets
    dataset_size = len(train_dataset)
    train_size = int(dataset_size * train_val_split)
    val_size = dataset_size - train_size
    train_set, val_set = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Ensure reproducibility
    )

    # Create the data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print("Contents of test_path:", os.listdir(test_path))
    print("Contents of train_path:", os.listdir(train_path))

    # Load the test dataset
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test directory {test_path} does not exist.")
    test_set = ImageFolder(root=test_path, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader. test_set


def compute_accuracy(y_pred, y_batch):
    accy = (y_pred == y_batch).sum().item() / y_batch.size(0)
    return accy


def main():
    # get the device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.cuda.set_device(device=0)
    torch.cuda.empty_cache()
    print("device: ", device)
    if use_cuda:
        torch.cuda.manual_seed(72)

    train_loader, test_loader, test_set = load_data('./asl-alphabet/versions/1', args.batch_size)

    # Your training code here
    model = CNNModel(args)

    ## load model to gpu or cpu
    model.to(device)

    ## initialize hyper-parameters
    num_epoches = args.num_epoches
    learning_rate = args.learning_rate

    ## define criterion, optimizer, and scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                           amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=0)

    # Get data loaders
    if args.mode == 'train':
        model.train()

        for epoch in range(num_epoches):
            print(f"\nEpoch {epoch}/{num_epoches}")
            print("-" * 20)

            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # Move to device
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += labels.size(0)
                epoch_correct += (predicted == labels).sum().item()

                # Batch-level logging
                batch_acc = (predicted == labels).float().mean().item()
                print(f"Batch {batch_idx}: Loss = {loss.item():.4f}, Batch Accuracy = {batch_acc:.4f}")

                # Accumulate epoch loss
                epoch_loss += loss.item()

            # Epoch-level metrics
            epoch_loss /= len(train_loader)
            epoch_accuracy = epoch_correct / epoch_total

            print(f"Epoch {epoch} Summary:")
            print(f"  Average Loss: {epoch_loss:.4f}")
            print(f"  Epoch Accuracy: {epoch_accuracy:.4f}")

            # WandB logging
            wandb.log({
                'epoch': epoch,
                'loss': epoch_loss,
                'accuracy': epoch_accuracy
            })

            # Learning rate scheduling
            scheduler.step(epoch_loss)

    test_acc = 0.0

    model.eval()
    pred_vec = []
    correct = 0

    with torch.no_grad():
        for data in test_loader:
            x_batch, y_labels = data
            x_batch, y_labels = x_batch.to(device), y_labels.to(device)

            output_y = model(x_batch)
            _, predicted = torch.max(output_y, 1)

            correct += (predicted == labels).sum().item()
            pred_vec.append(predicted)
        pred_vec = torch.cat(pred_vec)

    print("test accuracy: ", test_acc / len(test_loader))

    # visualize wrongly classified image for each class
    pred_vec = pred_vec.cpu().numpy()
    ground_truths = np.asarray(test_set.targets)
    incorrect_mask = pred_vec != ground_truths
    incorrect_images = [test_set.data[(ground_truths == label) & incorrect_mask][0] for label in range(10)]
    pred_results = [pred_vec[(ground_truths == label) & incorrect_mask][0] for label in range(10)]

    # show images
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    i = 0
    for row in axes:
        for axis in row:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlabel("Predicted: %s" % classes[pred_results[i]], fontsize=14)
            axis.imshow(incorrect_images[i])
            i += 1

if __name__ == '__main__':
    with wandb.init(project='ASL', name='ASL Project'):
        time_start = time.time()
        main()
        time_end = time.time()
        print("running time: ", (time_end - time_start) / 60.0, "mins")
