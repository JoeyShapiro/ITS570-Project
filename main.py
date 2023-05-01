from graph import build_hetero_graph
from torch_geometric.utils.convert import from_networkx

# g, connections = build_hetero_graph('test.pcap')
# pyg_graph = from_networkx(g)
# print(pyg_graph)

from dataset import Networks

dataset = Networks(root='/tmp/Networks', name='test')

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, SAGEConv, Linear, GATConv

class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(dataset.num_node_features, dataset.num_edge_features)
        self.conv2 = GATConv(dataset.num_edge_features, int(dataset.num_features))
        self.conv3 = GATConv(int(dataset.num_features), 2)
        self.classifier = Linear(2, 21)

    def forward(self, data):
        x, edge_index, edge_attr = torch.Tensor(data.x), data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.classifier(x)

        return F.log_softmax(x, dim=0)

#Since we have one graph in the dataset, we will select the graph and explore it's properties
print('Dataset properties')
print('==============================================================')
print(f'Dataset: {dataset}') #This prints the name of the dataset
print(f'Number of graphs in the datasetset: {len(dataset)}')
print(f'Number of edge features: {dataset.num_edge_features}')
print(f'Number of features: {dataset.num_features}') #Number of features each node in the dataset has
print(f'Number of classes: {dataset.num_classes}') #Number of classes that a node can be classified into

# Gather some statistics about the graph.
print('Graph properties')
print('==============================================================')
print(f'Data: {dataset[0]}')
print(f'Number of nodes: {dataset[0].num_nodes}') #Number of nodes in the graph # type: ignore
print(f'Number of edges: {dataset[0].num_edges}') #Number of edges in the graph # type: ignore
print(f'Number of edge features: {dataset[0].num_edge_features}')
print(f'Average node degree: {dataset[0].num_edges / dataset[0].num_nodes:.2f}') # Average number of nodes in the graph # type: ignore
print(f'Contains isolated nodes: {dataset[0].has_isolated_nodes()}') #Does the graph contains nodes that are not connected # type: ignore
print(f'Contains self-loops: {dataset[0].has_self_loops()}') #Does the graph contains nodes that are linked to themselves # type: ignore
# print(f'Is undirected: {dataset[0].is_undirected()}') # type: ignore #Is the graph an undirected graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN().to(device)
data = dataset[0].to(device) # type: ignore
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
tepoch = 5
for epoch in range(1, tepoch+1):
    optimizer.zero_grad()
    out = model(data) # this calls forward
    loss = F.l1_loss(out[data.train_mask], data.y[data.train_mask]) # nil loss
    # loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch}/{tepoch}, Loss: {loss}')

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
