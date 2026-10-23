# %% [markdown]
# Source notebook: Common Task 2 Quark-Gluon Classification M2 CNN TF.ipynb
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
# # Model 2: Deep Learning

# %% [markdown]
# ## Importing Libraries

# %%
# Notebook cell 3
get_ipython().run_line_magic('pip', 'install pydot graphviz')

import os
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import tensorflow as tf
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Add, Input

# %% [markdown]
# ## Disable Warnings

# %%
# Notebook cell 5
import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

# %% [markdown]
# ## Load the Dataset

# %%
# Notebook cell 7
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
# Notebook cell 8
df1 = read_file('/kaggle/input/common-task-2-dataset/Task - 2 Data (Parquet)/jet0_run0.test.snappy.parquet')
df2 = read_file('/kaggle/input/common-task-2-dataset/Task - 2 Data (Parquet)/jet0_run1.test.snappy.parquet')
df3 = read_file('/kaggle/input/common-task-2-dataset/Task - 2 Data (Parquet)/jet0_run2.test.snappy.parquet')

# %% [markdown]
# ## Constructing X and y

# %%
# Notebook cell 10
def construct_X_y(df):
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
# Notebook cell 11
# Assuming 'df' is the pandas DataFrame
X1, y1 = construct_X_y(df1)
X2, y2 = construct_X_y(df2)
X3, y3 = construct_X_y(df3)

# %%
# Notebook cell 12
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
# Notebook cell 13
x1 = np.load('X1.npy')
x2 = np.load('X2.npy')
x3 = np.load('X3.npy')

y1 = np.load('y1.npy')
y2 = np.load('y2.npy')
y3 = np.load('y3.npy')

# %%
# Notebook cell 14
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
# Notebook cell 15
# Loading the content of x.npy into the variable x
x = np.load('x.npy')

# Loading the content of y.npy into the variable y
y = np.load('y.npy')

# %% [markdown]
# ## CNN Model Creation

# %%
# Notebook cell 17
model = Sequential()

# Add the first convolutional layer with 125 filters, each with a 3x3 kernel, ReLU activation, and input shape of (125, 125, 3)
model.add(Conv2D(125, (3, 3), activation='relu', input_shape=(125, 125, 3)))

# Add BatchNormalization layer
model.add(BatchNormalization())

# Add MaxPooling layer with pool size (2, 2)
model.add(MaxPooling2D((2, 2)))

# Add Dropout layer with a dropout rate of 0.25
model.add(Dropout(0.25))

# Add the second convolutional layer with 64 filters and a 3x3 kernel, ReLU activation
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add BatchNormalization layer
model.add(BatchNormalization())

# Add MaxPooling layer with pool size (2, 2)
model.add(MaxPooling2D((2, 2)))

# Add Dropout layer with a dropout rate of 0.25
model.add(Dropout(0.25))

# Add the third convolutional layer with 128 filters and a 3x3 kernel, ReLU activation
model.add(Conv2D(128, (3, 3), activation='relu'))

# Add BatchNormalization layer
model.add(BatchNormalization())

# Add MaxPooling layer with pool size (2, 2)
model.add(MaxPooling2D((2, 2)))

# Add Dropout layer with a dropout rate of 0.25
model.add(Dropout(0.25))

# Flatten the output of the convolutional layers
model.add(Flatten())

# Add a fully connected Dense layer with 128 units and ReLU activation
model.add(Dense(128, activation='relu'))

# Add BatchNormalization layer
model.add(BatchNormalization())

# Add Dropout layer with a dropout rate of 0.5
model.add(Dropout(0.5))

# Add output layer with 1 unit and sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Define the optimizer (Adam) with a learning rate of 1e-3
optimizer = Adam(learning_rate=1e-3)

# Compile the model with binary crossentropy loss and accuracy metric
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# %%
# Notebook cell 18
model.summary()

# %%
# Notebook cell 19
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# %% [markdown]
# ## Data Splitting

# %%
# Notebook cell 21
xtrain, xtest, ytrain, ytest = train_test_split(x,
                                               y,
                                               test_size = 0.2,
                                               random_state = 42)

# %% [markdown]
# ## Model Training

# %%
# Notebook cell 23
history = model.fit(xtrain, 
                    ytrain, 
                    epochs=5, 
                    batch_size=8, 
                    validation_split=0.1, 
                    verbose=1)

# %% [markdown]
# ### Prediction Probability

# %%
# Notebook cell 25
y_pred_probs = model.predict(xtest)

# %% [markdown]
# ### ROC-AUC Score

# %%
# Notebook cell 27
roc_auc = roc_auc_score(ytest, y_pred_probs)
print(f'ROC AUC score: {roc_auc}')

# %%
# Notebook cell 28
# Calculate the false positive rate (false_pr), true positive rate (true_pr), and thresholds
false_pr, true_pr, thresholds = roc_curve(ytest, y_pred_probs)

# %% [markdown]
# ### Calculate the ROC-AUC Scores

# %%
# Notebook cell 30
roc_auc = auc(false_pr, true_pr)

# %% [markdown]
# ## Plotting ROC-AUC Curve

# %%
# Notebook cell 32
plt.figure(figsize = (10, 10))
lw = 2
plt.plot(false_pr, true_pr, color='darkorange', lw=lw, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='-.')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
