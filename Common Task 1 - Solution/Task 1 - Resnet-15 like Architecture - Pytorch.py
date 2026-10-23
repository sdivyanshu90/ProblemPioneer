# %% [markdown]
# Source notebook: Task 1 - Resnet-15 like Architecture - Pytorch.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# ## Importing Libraries

# %%
# Notebook cell 2
import os
import h5py
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc, roc_auc_score

# %% [markdown]
# ## Disable Warnings

# %%
# Notebook cell 4
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# %% [markdown]
# ## Load the data

# %%
# Notebook cell 6
electrons_data = h5py.File('/kaggle/input/task-1-dataset/Task 1 Data/Electron_dataset.hdf5', 'r')

X_dset_electrons = electrons_data['X']
y_dset_electrons = electrons_data['y']
X_electrons = np.array(X_dset_electrons)
y_electrons = np.array(y_dset_electrons)

print(X_electrons.shape)
print(y_electrons.shape)

photons_data = h5py.File('/kaggle/input/task-1-dataset/Task 1 Data/Photon_dataset.hdf5', 'r')

X_dset_photons = photons_data['X']
y_dset_photons = photons_data['y']
X_photons = np.array(X_dset_photons)
y_photons = np.array(y_dset_photons)

print(X_photons.shape)
print(y_photons.shape)

X = np.concatenate((X_electrons, X_photons), axis = 0)
y = np.concatenate((y_electrons, y_photons), axis = 0).reshape(-1,1)

print(X.shape)
print(y.shape)

# %% [markdown]
# ## Splitting the Data

# %%
# Notebook cell 8
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.015, random_state=42)

# %% [markdown]
# ## Convert data to PyTorch tensors

# %%
# Notebook cell 10
xtrain = torch.tensor(xtrain, dtype=torch.float32).permute(0, 3, 1, 2)
ytrain = torch.tensor(ytrain, dtype=torch.float32)
xtest = torch.tensor(xtest, dtype=torch.float32).permute(0, 3, 1, 2)
ytest = torch.tensor(ytest, dtype=torch.float32)

# %% [markdown]
# ## Create DataLoader objects

# %%
# Notebook cell 12
train_dataset = TensorDataset(xtrain, ytrain)
test_dataset = TensorDataset(xtest, ytest)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %% [markdown]
# ## ResNet - 15 like model Creation

# %%
# Notebook cell 14
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# %%
# Notebook cell 15
class ResNet15(nn.Module):
    def __init__(self):
        super(ResNet15, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 64, stride=2)
        self.res3 = ResidualBlock(64, 128, stride=2)

        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# %% [markdown]
# ## Creating Model

# %%
# Notebook cell 17
def create_model():
    model = ResNet15()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    return model, criterion, optimizer

# %% [markdown]
# ## Device configuration

# %%
# Notebook cell 19
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [markdown]
# ## Model, criterion, and optimizer

# %%
# Notebook cell 21
model, criterion, optimizer = create_model()
model = model.to(device)

# %% [markdown]
# ## Training

# %%
# Notebook cell 23
def train(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(train_loader, desc="Training: "):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = torch.round(torch.sigmoid(outputs))

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    return epoch_loss, epoch_acc

# %% [markdown]
# ## Testing

# %%
# Notebook cell 25
def test(model, criterion, test_loader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    true_labels = []
    predicted_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing: "):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.round(torch.sigmoid(outputs))

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

            true_labels.extend(labels.cpu().numpy().flatten())
            predicted_probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())

        test_loss = running_loss / len(test_loader.dataset)
        test_acc = running_corrects.double() / len(test_loader.dataset)
        roc_auc = roc_auc_score(true_labels, predicted_probs)

    return test_loss, test_acc, roc_auc, true_labels, predicted_probs

# %% [markdown]
# ## Plotting ROC Curve

# %%
# Notebook cell 27
def plot_roc_curve(false_pr, true_pr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(false_pr, true_pr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='-.')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

# %% [markdown]
# ## Function for Training the Model

# %%
# Notebook cell 29
def run_training(model, criterion, optimizer, train_loader, test_loader, device, num_epochs = 50):
    all_true_labels = []
    all_predicted_probs = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, criterion, optimizer, train_loader, device)
        test_loss, test_acc, roc_auc, true_labels, predicted_probs = test(model, criterion, test_loader, device)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, ROC AUC: {roc_auc:.4f}')

        # Accumulate true labels and predicted probabilities
        all_true_labels.extend(true_labels)
        all_predicted_probs.extend(predicted_probs)

    # Calculate the false positive rate, true positive rate, and thresholds
    false_pr, true_pr, _ = roc_curve(all_true_labels, all_predicted_probs)

    # Calculate the ROC AUC
    roc_auc = auc(false_pr, true_pr)

    # Plot the ROC curve
    plot_roc_curve(false_pr, true_pr, roc_auc)

run_training(model, criterion, optimizer, train_loader, test_loader, device)
