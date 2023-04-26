from graph import build_hetero_graph
from torch_geometric.utils.convert import from_networkx

g, connections = build_hetero_graph('test.pcap')
pyg_graph = from_networkx(g)
print(pyg_graph)
# print(pyg_graph['features'], pyg_graph['labels'])

# from torch_geometric.datasets import Planetoid, KarateClub
from dataset import Networks
from torch_geometric.datasets import Planetoid

dataset = Networks(root='/tmp/Networks', name='test')
# dataset = Planetoid(root='/tmp/Cora', name='Cora')
print(type(dataset))

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, SAGEConv

class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        print('edge features: ', dataset.num_edge_features)
        # self.conv1 = TransformerConv(dataset.num_edge_features, 16)
        self.conv1 = GCNConv(5, 4) # TODO this might help the interval [0 0] problrm
        self.conv2 = GCNConv(4, int(dataset.num_classes)) # something about 17 and 5

    def forward(self, data):
        x, edge_index = torch.Tensor(data.x), data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=0)

from torch_geometric.datasets import KarateClub

# dataset = KarateClub()
data = dataset[0]
print(type(data))
print('Dataset properties')
print('==============================================================')
print(f'Dataset: {dataset}') #This prints the name of the dataset
print(f'Number of graphs in the dataset: {len(dataset)}')
print(f'Number of features: {dataset.num_features}') #Number of features each node in the dataset has
print(f'Number of classes: {dataset.num_classes}') #Number of classes that a node can be classified into


#Since we have one graph in the dataset, we will select the graph and explore it's properties

data = dataset[0]
print('Graph properties')
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}') #Number of nodes in the graph
print(f'Number of edges: {data.num_edges}') #Number of edges in the graph
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}') # Average number of nodes in the graph
print(f'Contains isolated nodes: {data.has_isolated_nodes()}') #Does the graph contains nodes that are not connected
print(f'Contains self-loops: {data.has_self_loops()}') #Does the graph contains nodes that are linked to themselves
print(f'Is undirected: {data.is_undirected()}') #Is the graph an undirected graph

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        
        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h

model = GCN()
criterion = torch.nn.CrossEntropyLoss()  #Initialize the CrossEntropyLoss function.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Initialize the Adam optimizer.

def train(data):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h

for epoch in range(401):
    loss, h = train(data)
    print(f'Epoch: {epoch}, Loss: {loss}')

print(type(dataset), type(dataset[0]))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN().to(device)
data = dataset[0].to(device) # type: ignore
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
tepoch = 1
for epoch in range(tepoch): # TODO something is happening, just create new set like karate
    print(f'epoch: {epoch}/{tepoch}')
    optimizer.zero_grad()
    out = model(data) # this calls forward
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    # loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')

# try with networkx, but may need to use pytorch. at least its standardized
# from torch.nn import Linear, ReLU
# from torch_geometric.nn import Sequential, GCNConv, TransformerConv

# model = Sequential('x, edge_index', [
#     (GCNConv(in_channels, 64), 'x, edge_index -> x'),
#     ReLU(inplace=True),
#     (GCNConv(64, 64), 'x, edge_index -> x'),
#     ReLU(inplace=True),
#     Linear(64, out_channels),
# ])
