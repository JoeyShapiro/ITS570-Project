import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, SAGEConv, Linear, GATConv, GINEConv

class GNN(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GATConv(dataset.num_node_features, dataset.num_edge_features)
        self.conv2 = GATConv(dataset.num_edge_features, int(dataset.num_features))
        self.conv3 = GATConv(int(dataset.num_features), dataset.num_classes)
        self.classifier = Linear(dataset.num_classes, dataset.num_node_features) # this 21 was causing issues

    def forward(self, data):
        x, edge_index, edge_attr = torch.Tensor(data.x), data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.classifier(x)

        return F.log_softmax(x, dim=0)