# %% [markdown]
# Source notebook: Task 3g GAT model.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

from IPython import get_ipython as _active_ipython

def get_ipython():
    """Use the active shell and expose this script namespace to line magics."""
    shell = _active_ipython()
    if shell is None:
        raise RuntimeError("This source uses IPython commands; run it with ipython instead of python.")
    import inspect
    caller = inspect.currentframe().f_back
    shell.user_ns.update(caller.f_globals)
    shell.user_ns.update(caller.f_locals)
    return shell

# %% [markdown]
# ## Import Libraries

# %%
# Notebook cell 2
get_ipython().system('pip install torch-geometric')

import os
import numpy as np
import pyarrow as pa
import pandas as pd
import pyarrow.parquet as pq
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib.pyplot as plt

# %% [markdown]
# ## Disable Warnings

# %%
# Notebook cell 4
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# %% [markdown]
# ## Load Data

# %%
# Notebook cell 6
def read_file(path):
  chunk_size = 25

# Create a Parquet file reader object
  parquet_file = pq.ParquetFile(path)

# Determine the total number of rows in the file
  total_rows = parquet_file.metadata.num_rows

# Loop over the file in chunks
  data = []
  for i in range(0, total_rows, chunk_size):
    # Read a chunk of rows from the file
      chunk = (parquet_file.read_row_group(i))
      dm = (chunk.to_pandas())
      data.append(dm)

# Concatenate all the DataFrames into a single DataFrame
  df = pd.concat(data, ignore_index=True)
  print(parquet_file.read_row_group(0).to_pandas())
  return df

# %%
# Notebook cell 7
df1 = read_file('/kaggle/input/common-task-2-dataset/Task - 2 Data (Parquet)/jet0_run0.test.snappy.parquet')
df2 = read_file('/kaggle/input/common-task-2-dataset/Task - 2 Data (Parquet)/jet0_run1.test.snappy.parquet')
df3 = read_file('/kaggle/input/common-task-2-dataset/Task - 2 Data (Parquet)/jet0_run2.test.snappy.parquet')

# %%
# Notebook cell 8
def construct_image_dataset(df):
    # Get the total number of samples
    num_samples = len(df)

    # Initialize empty arrays for X and y
    X = np.empty((num_samples, 3, 125, 125), dtype=np.float32)
    y = np.empty(num_samples, dtype=int)

    # Iterate through the DataFrame and fill X and y
    for i, row in df.iterrows():
        # Stack the three channels of X_jets and transpose them to match the desired shape
        X[i] = np.transpose(np.dstack((np.stack(row['X_jets'][0]), np.stack(row['X_jets'][1]), np.stack(row['X_jets'][2]))), (2, 0, 1))
        # Assign the label to y
        y[i] = row['y']

    # Rearrange the dimensions of X to match the TensorFlow format (samples, height, width, channels)
    X = np.transpose(X, (0, 2, 3, 1))

    return X, y

# %%
# Notebook cell 9
# Assuming 'df' is the pandas DataFrame
X1, y1 = construct_image_dataset(df1)
X2, y2 = construct_image_dataset(df2)
X3, y3 = construct_image_dataset(df3)

# %%
# Notebook cell 10
# Save X1 array to binary file 'X1.npy'
with open('X1.npy', 'wb') as f:
    np.save(f, X1)

# Save y1 array to binary file 'y1.npy'
with open('y1.npy', 'wb') as f:
    np.save(f, y1)

# Save X2 array to binary file 'X2.npy'
with open('X2.npy', 'wb') as f:
    np.save(f, X2)

# Save y2 array to binary file 'y2.npy'
with open('y2.npy', 'wb') as f:
    np.save(f, y2)

# Save X3 array to binary file 'X3.npy'
with open('X3.npy', 'wb') as f:
    np.save(f, X3)

# Save y3 array to binary file 'y3.npy'
with open('y3.npy', 'wb') as f:
    np.save(f, y3)

# %%
# Notebook cell 11
x1 = np.load('X1.npy')
x2 = np.load('X2.npy')
x3 = np.load('X3.npy')

y1 = np.load('y1.npy')
y2 = np.load('y2.npy')
y3 = np.load('y3.npy')

# %%
# Notebook cell 12
# Combine x arrays along the first axis (axis=0)
x = np.concatenate((x1, x2, x3), axis=0)

# Combine y arrays along the first axis (axis=0)
y = np.concatenate((y1, y2, y3), axis=0)

# Save the combined X array to a binary file named 'X.npy'
with open('x.npy', 'wb') as f:
    np.save(f, x)

# Save the combined y array to a binary file named 'y.npy'
with open('y.npy', 'wb') as f:
    np.save(f, y)

# %%
# Notebook cell 13
# Load the dataset
X = np.load("x.npy")
y = np.load("y.npy")

# Convert the numpy arrays to PyTorch tensors
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.int)

# %%
# Notebook cell 14
def linear_kernel(distance_matrix, gamma=None):
    if gamma is None:
        gamma = 1.0 / distance_matrix.shape[1]
    return gamma * np.dot(distance_matrix, gamma)

def construct_graph_using_linear_kernel(image, label):
    height, width, channels = image.shape
    nodes = []
    positions = []

    # Iterate over the 5x5 crops and extract features
    for i in range(0, height, 5):
        for j in range(0, width, 5):
            crop = image[i:i+5, j:j+5]
            feature = crop.reshape(-1)
            nodes.append(feature)
            positions.append((i, j))

    nodes = torch.stack(nodes)
    positions = np.array(positions)

    # Compute edge_index and edge_attr using Linear Kernel
    distance_matrix = cdist(positions, positions)
    linear_values = linear_kernel(distance_matrix)
    edges = np.where(linear_values > 0.8)  # Adjust threshold as needed
    edge_index = torch.tensor(np.vstack(edges), dtype=torch.long)
    edge_attr = torch.tensor(linear_values[edges], dtype=torch.float)

    # Create the PyTorch Geometric Data object
    data = Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label], dtype=torch.long))
    return data

# %%
# Notebook cell 15
filename = "pyg_data_list.pkl"
if os.path.exists(filename):
    print(f"{filename} exists in the current directory.")
    with open("pyg_data_list.pkl", "rb") as file:
        pyg_data_list = pickle.load(file)
else:
    print(f"{filename} does not exist in the current directory.")
    # Create PyTorch Geometric Data objects for each image
    pyg_data_list = [construct_graph_using_linear_kernel(X[i], y[i]) for i in tqdm(range(X.shape[0]), desc="Creating Data Objects")]

# %% [markdown]
# ## Splitting Data into train_dataset and test_dataset

# %%
# Notebook cell 17
# Set up the train and test datasets
train_dataset = pyg_data_list[:int(len(pyg_data_list) * 0.75)]
test_dataset = pyg_data_list[int(len(pyg_data_list) * 0.75):]

# Set up the data loader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %% [markdown]
# ## Building Graph Attention Networks (GAT) Model

# %%
# Notebook cell 19
class GATModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        return x

# %% [markdown]
# ## Set up the GAT model, loss function, and optimizer

# %%
# Notebook cell 21
num_node_features = train_dataset[0].x.shape[1]
hidden_channels = 64
num_classes = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GATModel(num_node_features, hidden_channels, num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# %% [markdown]
# ## Define training and evaluation functions

# %%
# Notebook cell 23
def train():
    model.train()
    total_loss = 0
    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad() # To Clear gradients
        out = model(data) # Perform a single forward pass
        loss = criterion(out, data.y) # Compute the loss
        loss.backward() # Deriving gradients
        optimizer.step() # Updating parameters
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(loader):
    model.eval()
    correct = 0
    y_true = []
    y_probs = []

    for data in tqdm(loader, desc="Testing"):
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        probs = F.softmax(out, dim=-1)
        pred = out.argmax(dim=-1)

        y_true.extend(data.y.tolist())
        y_probs.extend(probs[:, 1].tolist())

        correct += int((pred == data.y).sum())

    false_pr, true_pr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    accuracy = correct / len(loader.dataset)

    return accuracy, false_pr, true_pr, roc_auc

# %% [markdown]
# ## Function to plot ROC curve

# %%
# Notebook cell 25
def plot_roc_curve(false_pr, true_pr, roc_auc, title="ROC curve"):
    plt.figure()
    plt.plot(false_pr, true_pr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle="-.")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig("gat-roc-curve.png")
    plt.show()

# %% [markdown]
# ## Train and Evaluate the model

# %%
# Notebook cell 27
num_epochs = 50
max_auc = 0

for epoch in range(num_epochs):
    loss = train()
    train_acc, _, _, train_roc_auc = test(train_loader)
    test_acc, false_pr, true_pr, test_roc_auc = test(test_loader)
    print(f"Epoch: {epoch + 1}, Loss: {loss:.4f}, Train (Accuracy): {train_acc:.4f}, Train (ROC-AUC): {train_roc_auc:.4f}, Test (Accuracy): {test_acc:.4f}, Test (ROC-AUC): {test_roc_auc:.4f}")
    if test_roc_auc > max_auc:
        max_auc = test_roc_auc
        max_false_pr, max_true_pr, = false_pr, true_pr

plot_roc_curve(max_false_pr, max_true_pr, max_auc, title="ROC curve (Test set)")
