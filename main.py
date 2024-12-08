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
parser.add_argument("-num_epoches", dest="num_epoches", type=int, default=1, help="num of epoches")

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
    transforms.Resize((150, 150)),  # Resize the images to 150x150 pixels.
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
    live_test_path = os.path.join(data_dir, 'LiveActionConverted')

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

    # Load the test dataset
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test directory {test_path} does not exist.")
    test_set = ImageFolder(root=test_path, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    #Load live test dataset
    if not os.path.exists(live_test_path):
        raise FileNotFoundError(f"Live test directory {live_test_path} does not exist.")
    live_test_set = ImageFolder(root=live_test_path, transform=test_transform)
    live_test_loader = DataLoader(live_test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader, test_set, live_test_loader, live_test_set


def compute_accuracy(y_pred, y_batch):
    accy = (y_pred == y_batch).sum().item() / y_batch.size(0)
    return accy


def visualize_incorrect_predictions(pred_vec, set, set_loader):

    # Find incorrect images for each class
    incorrect_images = []
    incorrect_preds = []

    ground_truths = torch.tensor(set.targets)
    pred_vec = pred_vec.cpu().numpy()
    incorrect_mask = pred_vec != ground_truths

    for label in range(29):
        # Find indices of incorrectly classified images for this label
        class_incorrect_indices = np.where((ground_truths.numpy() == label) & incorrect_mask)[0]
        assert np.all(np.logical_or(incorrect_mask, ~incorrect_mask)), "Incorrect mask must be boolean."

        if len(class_incorrect_indices) > 0:
            incorrect_index = class_incorrect_indices[0]

            incorrect_image, true_label = set_loader.dataset[incorrect_index]
            incorrect_image = (incorrect_image * 255).permute(1, 2, 0).numpy().astype(np.uint8)

            incorrect_images.append(incorrect_image)
            incorrect_preds.append(pred_vec[incorrect_index])
        else:
            print(f"No incorrect images for class {label}. Adding placeholder.")
            incorrect_images.append(np.zeros((64, 64, 3), dtype=np.uint8))
            incorrect_preds.append(label)

    # Visualize
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

    print(f"pred_vec shape: {pred_vec.shape}, ground_truths shape: {ground_truths.shape}")
    print(f"Number of incorrect images: {len([img for img in incorrect_images if np.any(img)])}")

    # Filter out only incorrect predictions
    filtered_images = []
    for idx in range(len(ground_truths)):
        true_label = classes[ground_truths[idx]]
        pred_label = classes[incorrect_preds[idx]]
        if true_label != pred_label:
            incorrect_image = incorrect_images[idx]
            filtered_images.append((incorrect_image, true_label, pred_label))

    # Handle case where there are no incorrect predictions
    if len(filtered_images) == 0:
        print("No incorrect predictions to display.")
        exit()

    # Visualize only the filtered incorrect predictions
    num_images = len(filtered_images)
    # Calculate the grid size
    cols = 4  # Number of columns
    rows = (num_images + cols - 1) // cols  # Calculate rows dynamically based on the number of images

    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))

    # Flatten axes for easier indexing, in case of multiple rows and columns
    axes = axes.flatten() if num_images > 1 else [axes]

    for i in range(num_images):
        img, true_label, pred_label = filtered_images[i]
        axes[i].imshow(img)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)

    # Turn off any unused axes
    for j in range(num_images, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def main():
    # get the device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.cuda.set_device(device=0)
    torch.cuda.empty_cache()
    print("device: ", device)
    if use_cuda:
        torch.cuda.manual_seed(72)

    train_loader, test_loader, test_set, live_loader, live_set = load_data('./asl-alphabet/versions/1', args.batch_size)

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

            print(f"Epoch {epoch + 1} Summary:")
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
    live_acc = 0.0

    model.eval()
    pred_vec = []
    live_vec = []

    with torch.no_grad():
        #dataset test data
        for data in test_loader:
            x_batch, y_labels = data
            x_batch, y_labels = x_batch.to(device), y_labels.to(device)

            output_y = model(x_batch)
            y_pred = torch.argmax(output_y.data, 1)

            test_acc += compute_accuracy(y_pred, y_labels)
            pred_vec.append(y_pred)

        # live test data
        for data in live_loader:
            x_batch, y_labels = data
            x_batch, y_labels = x_batch.to(device), y_labels.to(device)

            output_y = model(x_batch)
            y_pred = torch.argmax(output_y.data, 1)

            live_acc += compute_accuracy(y_pred, y_labels)
            live_vec.append(y_pred)

        pred_vec = torch.cat(pred_vec)
        live_vec = torch.cat(live_vec)

    print("dataset test accuracy: ", test_acc / len(test_loader))
    print("live test accuracy: ", live_acc / len(live_loader))

    visualize_incorrect_predictions(pred_vec, test_set, test_loader)
    visualize_incorrect_predictions(live_vec, live_set, live_loader)



if __name__ == '__main__':
    with wandb.init(project='ASL', name='ASL Project'):
        time_start = time.time()
        main()
        time_end = time.time()
        print("running time: ", (time_end - time_start) / 60.0, "mins")
