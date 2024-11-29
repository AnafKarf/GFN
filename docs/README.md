<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

# How to predict if a molecule is biotherapeutic using Graph Feature Network

## Authors
- Sofia Tkachenko, Innopolis University, B22-DS-02
- Anastasiia Shvets, Innopolis University, B22-DS-02
- David Khachikov, Innopolis University, B22-DS-02

## Introduction

Currently biotherapeutic drugs are gaining popularity. 
However, such molecules can be harder to test: you have to use living organisms to test them, while non-biotherapeutic medicine can be tested using computer simulations. 
Thus, it is important to distinguish such molecules before testing them. One of the ways to do it is using Graph Feature Network.

## What is a Graph and why can we use it for molecules?
![Example of a graph](/GFN/assets/graph.png)
*Example of a graph*

A graph is a structure that shows connection between set of objects. 

A graph consists of two main parts:
- Nodes (Vertices) - units of a graph, represent some object. 
- Edges (Links) - connections between these units, represent the relationships between the objects.

Both nodes and edges may have different types and features.

The reason why we represent molecules as graphs is because they also consist of units (atoms) and connections between those units. 
![Example of a molecule](/GFN/assets/molekula-glicina.jpg)
*Example of a molecule*

## What is a Graph Feature Network?
Do you know what a Convolutional Neural Network is? 
Well, Graph Feature Network is a modification of Graph Convolutional Network, which is similar to Convolutional Neural Network. 
The purpose of GCN is to capture the features of graphs and create a similarity function in such a way that similar graphs are embedded close together. 
![How graph embedding works](/GFN/assets/graph_embeddings.png)
*Our goal, when creating networks on top of graphs*

So, let's start from the beginning!

### GCN architecture
To compare the new solution (Graph Feature Network), it is first necessary to understand how Graph Convolutional Networks (GCN) function. 
Similar to Convolutional Neural Networks (CNN), GCN aim to capture both local and global information in order to perform node prediction tasks. 
Additionally, if a node-level task can be solved, the graph-level task can easily be addressed by applying any aggregation function to the node embeddings, such as a Multi-Layer Perceptron (MLP), sum, or average.

![GCN](/GFN/assets/gcn_web.png)
*Graph Convolutional Network*

#### Initialization
Randomly assign embedding to each node or give the same vector for every node.

#### Graph convolutional layer
To introduce GCN, we need to define a few objects that we will use to build the network on top of. The normalized adjacency matrix is defined by the following formula: 

$$accent(A, hat) = D^{-\frac{1}{2}}A D^{-\frac{1}{2}}$$ 

Here, $A$ is the square adjacency matrix for the given graph and $D$ is a diagonal matrix with the same dimensions as $A$. 
The $D_{i, i}$ corresponds to the degree of the *i*th node. Why do we need to normalize the matrix? The reason is that we don't want outliers to overshadow other information about the graph topology. 
Some of our team members actually encountered such problem, when they were working to create a recommendation system for student accommodation. 
In that case, several students agreed to live with anyone, and if the distribution of freshmen into rooms was performed without normalization, the loss function would be dramatically unstable.


The graph convolutional layer is the core building block of GCNs. It updates each node's embedding by aggregating information from its neighbors, weighted by the graph's adjacency structure. 
This ensures that a node's new representation reflects both its own features and the features of its local neighborhood. Mathematically, this can be described as:

$$H^{l + 1} = \sigma(\tilde{A}H^l W^l)$$

where:
    - $H^l$ is the node embedding matrix at layer l,
    - $\tilde{A}$ is the normalized adjacency matrix,
    - $W^l$ is the learnable weight matrix at layer l,
    - $\sigma$ is a non-linear activation function (e.g., ReLU).

### Difference between GFN (Graph Feature Network) and GCN

The main advantage of GFN (Graph Feature Network) over GCN is its simplicity in terms of the filtering function. 
This is achieved by linearizing the graph-based filtering function, which creates a lightweight neural network that is defined on a set of augmented graph features. 
If our network is even more lightweight, we can simply stack additional layers, allowing for parallel learning.

$$X^G = [d, X, \tilde{A}^1X, \tilde{A}^2X, ..., \tilde{A}^K X]$$

where:
  - d is the vector of node degrees.
  - X is the features vector.

Basically $X^G$ if the feature vector of the whole graph, which is obtained by multiplying the normalized adjacency matrix by different powers and then concatenating the results.
  
Then GFN convolution is defined as follows:

$$GFN(G, X) = \rho(\sum_{v \in V}\phi(X^G_v))$$

where $\rho$ and $\phi$ are fully connected neural networks.

With such additions, computational complexity is reduced by performing neighborhood aggregation before the training even starts.

## Implementation details

### Install and import necessary libraries
For the project you will need to install the following libraries:
- kagglehub
- matplotlib
- numpy
- pandas
- rdkit
- sklearn
- torch
- torch-geometric
- torch-scatter

Make sure that you install torch-scatter and torch-geometric compatible with your torch version!

Here are all imports you will need to use:

```python
  import ast
  import os
  
  from functools import partial
  
  import kagglehub
  import rdkit
  import torch
  
  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd
  import torch.nn.functional as F

  from rdkit import Chem
  from rdkit.Chem import AllChem
  from rdkit.Chem.rdmolops import GetAdjacencyMatrix

  from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
  
  from torch.nn import Parameter, Linear, BatchNorm1d
  
  from torch_geometric.data import Data, DataLoader, Dataset
  from torch_geometric.nn import global_add_pool, global_mean_pool
  from torch_geometric.nn.conv import MessagePassing
  from torch_geometric.nn.inits import glorot, zeros
  from torch_geometric.utils import add_self_loops, remove_self_loops

  from torch_scatter import scatter_add
```

### Load dataset
In this example, we use our custom dataset collected from [ChEMBL](https://www.ebi.ac.uk/chembl/)[ChEMBL]. 
You can view the dataset [on Kaggle](https://www.kaggle.com/datasets/davidhachikov/biotherapeutic-property-dataset). 

First, load the dataset you want to use:

```python
  path = kagglehub.dataset_download("davidhachikov/biotherapeutic-property-dataset")

  print("Path to dataset files:", path)
```

In our case, the dataset is already divided into train and test parts. 
However, you may need to divide the dataset into train and test parts yourself, if you want to use another one. 

Since in the dataset each molecule is represented in the SMILES format, we need to convert it to graph. 
You can follow [this tutorial](https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/)[this tutorial] to know how to do it. 
However, for our convenience, we wrap the dataset into ```python SmilesDataset``` class. 
It uses ```python create_pytorch_geometric_graph_data_list_from_smiles_and_labels``` function from the tutorial, and inherits from ```python Dataset``` class defined in ```python torch_geometric.data```

```python
class SmilesDataset(Dataset):
    def __init__(self, dataframe, transform=None, pre_transform=None):
        super(SmilesDataset, self).__init__(None, transform, pre_transform)
        self.dataframe = dataframe

        self.graphs = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(
            self.dataframe['smiles'].to_list(),
            self.dataframe['toxicity'].to_list()
        )

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        graph = self.graphs[idx]
        if graph is None:
            raise ValueError(f"Invalid SMILES string at index {idx}")
        return graph
```

```python
def preprocess(csv_path):
    df = pd.read_csv(csv_path)

    # Rename columns with meaningful names
    df.columns = ['molecule_chembl_id', 'molecule_structures', 'biotherapeutic']

    # Extract canonical_smiles from molecule_structures
    df['canonical_smiles'] = df['molecule_structures'].apply(lambda x: ast.literal_eval(x)['canonical_smiles'])

    # Create a new target column that is True if either oral or topical is True
    df['target'] = df['biotherapeutic']

    return SmilesDataset(df)
```

Create train dataset and dataloader using the functions defined above.

```python
  train_dataset = preprocess(f'{path}/data/train/molecules_train.csv')
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

Similarly define the loader for test data.

### Define the model

#### Convolutional block
Before creating the model itself, we need to define the convolutional block:

```python
class GCNConv(MessagePassing):
def __init__(self,
            in_channels,
            out_channels,
            gfn=True):
    super(GCNConv, self).__init__('add')

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.gfn = gfn

    self.weight = Parameter(torch.Tensor(in_channels, out_channels))

    self.bias = Parameter(torch.Tensor(out_channels))

    self.reset_parameters()

def reset_parameters(self):
    glorot(self.weight)
    zeros(self.bias)

@staticmethod
def norm(edge_index, num_nodes, edge_weight, dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ),
                                  dtype=dtype,
                                  device=edge_index.device)
    edge_weight = edge_weight.view(-1)
    assert edge_weight.size(0) == edge_index.size(1)

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    # Add edge_weight for loop edges.
    loop_weight = torch.full((num_nodes, ),
                              1,
                              dtype=edge_weight.dtype,
                              device=edge_weight.device)
    edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def forward(self, x, edge_index, edge_weight=None):
    """"""
    x = torch.matmul(x, self.weight)
    if self.gfn:
        return x

    edge_index, norm = GCNConv.norm(
        edge_index, x.size(0), edge_weight, x.dtype)

    return self.propagate(edge_index, x=x, norm=norm)

def message(self, x_j, norm):
    return norm.view(-1, 1) * x_j

def update(self, aggr_out):
    if self.bias is not None:
        aggr_out = aggr_out + self.bias
    return aggr_out
```

Key Components:

    - ```python self.weight```: A learnable parameter matrix for feature transformation, initialized with the Glorot method. Glorot initialization for weights ensures the variance of the parameters remains stable during training.
    
    - ```python self.bias```: A learnable bias term, initialized to zeros.
    
    - ```python reset_parameters```: Resets the weights and biases to their initial values.

#### GFN model
Now, we can combine our custom convolution layer with other components to create a single model that includes all the latest techniques such as dropout, batch normalization, and residual networks. 
The code here is more technical and less graph-specific, which ensures the flexibility of boolean flag changes and the ability to stack layers, among other things.

```python
class GFN(torch.nn.Module):
    def __init__(
        self,
        dataset,
        hidden,
        num_feat_layers=1,
        num_conv_layers=3,
        num_fc_layers=2,
        dropout=0.3,
    ):
        super(GFN, self).__init__()
        self.global_pool = global_add_pool
        self.dropout = dropout

        hidden_in = dataset.num_features
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True)
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GCNConv(hidden, hidden, gfn=True))

        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, dataset.num_classes)

        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_

        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x = F.relu(lin(x_))
        x = self.bn_hidden(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
          return F.log_softmax(x, dim=-1)
```

### Train the model
Below you can find the training script. If you have understood everything above, then this part of the code is probably the most boring one. 
Here, we are simply performing backpropagation using optimizer steps. 
Note that we have used balanced cross-entropy as a loss function, because our dataset is slightly unbalanced.

```python
hidden_dim = 128
epochs = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GFN(dataset=train_dataset, hidden=hidden_dim, num_conv_layers=5, num_fc_layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

labels = [data.y for data in train_dataset]
class_counts = torch.bincount(torch.tensor(labels))
class_weights = 1.0 / class_counts.float()
class_weights = class_weights.to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss / len(train_loader))

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            val_loss += loss.item()

            # Accuracy calculation
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

    val_losses.append(val_loss / len(test_loader))
    val_accuracy = correct / total
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy*100:.2f}%")
```

### Evaluation
![Model metrics](/GFN/assets/metrics_gfn.png)
*Model metrics*
We trained our model for 100 epochs and saw that the validation accuracy and training loss were excellent at the beginning. 
However,  the instability of the validation loss was noticeable. 
Despite this, the model performed greatly on the validation set.

Here are the metrics we obtained on the test dataset.
![Classification report](/GFN/assets/scikit-report.png)
*Classification report*

## Code
You can check the [Google Colab notebook](https://colab.research.google.com/drive/1HHyogoqQYtZH53v97Mf5fXzwk8I6X5w8?usp=sharing)[Google Colab notebook] with all the code for yourself.

### Finally
So, this is how you can predict different molecular properties, such as biotherapeutic origin, using GFN! 
If you have any questions or comments, please feel free to leave them in the comment section.
