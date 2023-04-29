from graph import build_hetero_graph
from torch_geometric.utils.convert import from_networkx

g, connections = build_hetero_graph('test.pcap')
pyg_graph = from_networkx(g)
print(pyg_graph)
# print(pyg_graph['features'], pyg_graph['labels'])

# from torch_geometric.datasets import Planetoid, KarateClub
from dataset import Networks

dataset = Networks(root='/tmp/Networks', name='test')
# dataset = Planetoid(root='/tmp/Cora', name='Cora')
print(type(dataset))

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, SAGEConv, Linear

class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        print('edge features: ', dataset.num_edge_features)
        self.conv1 = TransformerConv(dataset.num_edge_features, 4)
        self.conv2 = GCNConv(4, int(dataset.num_classes))
        self.conv3 = GCNConv(int(dataset.num_classes), 2)
        self.classifier = Linear(2, dataset.num_classes)

        # self.conv1 = GCNConv(dataset.num_features, 4)
        # self.conv2 = GCNConv(4, 4)
        # self.conv3 = GCNConv(4, 2)
        # self.classifier = Linear(2, dataset.num_classes)

    def forward(self, data):
        x, edge_index = torch.Tensor(data.x), data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.classifier(x)

        return F.log_softmax(x, dim=0)

# TODO put this in display for dataset, only display first
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

print(type(dataset), type(dataset[0]))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN().to(device)
data = dataset[0].to(device) # type: ignore
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
tepoch = 5
for epoch in range(1, tepoch+1):
    optimizer.zero_grad()
    out = model(data) # this calls forward
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    # loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch}/{tepoch}, Loss: {loss}')

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
