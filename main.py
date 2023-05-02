from graph import build_hetero_graph
from torch_geometric.utils.convert import from_networkx
from gnn import GNN

# g, connections = build_hetero_graph('test.pcap')
# pyg_graph = from_networkx(g)
# print(pyg_graph)

from dataset import Networks
from torch_geometric.datasets import Planetoid, KarateClub
from torch_geometric.loader import DataLoader

# dataset = KarateClub()
# dataset = Planetoid(root='/tmp/Planetoid', name='Cora')
dataset = Networks(root='/tmp/Networks', name='test')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

import torch
import torch.nn.functional as F

#Since we have one graph in the dataset, we will select the graph and explore it's properties
print('Dataset properties')
print('==============================================================')
print(f'Dataset: {dataset}') #This prints the name of the dataset
print(f'Number of graphs in the datasetset: {len(dataset)}')
print(f'Number of features: {dataset.num_features}') #Number of features each node in the dataset has
print(f'Number of edge features: {dataset.num_edge_features}')
print(f'Number of node features: {dataset.num_node_features}')
print(f'Number of classes: {dataset.num_classes}') #Number of classes that a node can be classified into

# Gather some statistics about the graph.
print('Graph properties')
print('==============================================================')
print(f'Data: {dataset[0]}')
print(f'Number of nodes: {dataset[0].num_nodes}') #Number of nodes in the graph # type: ignore
print(f'Number of edges: {dataset[0].num_edges}') #Number of edges in the graph # type: ignore
print(f'Number of edge features: {dataset[0].num_edge_features}')
print(f'Number of node features: {dataset[0].num_node_features}')
print(f'Average node degree: {dataset[0].num_edges / dataset[0].num_nodes:.2f}') # Average number of nodes in the graph # type: ignore
print(f'Contains isolated nodes: {dataset[0].has_isolated_nodes()}') #Does the graph contains nodes that are not connected # type: ignore
print(f'Contains self-loops: {dataset[0].has_self_loops()}') #Does the graph contains nodes that are linked to themselves # type: ignore
# print(f'Is undirected: {dataset[0].is_undirected()}') # type: ignore #Is the graph an undirected graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(dataset).to(device)
data = dataset[0].to(device) # type: ignore
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
tepoch = 200
for epoch in range(1, tepoch+1):
    optimizer.zero_grad()
    out = model(data) # this calls forward
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch}/{tepoch}, Loss: {loss}')

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')

torch.save(model.state_dict(), "model.pt")
