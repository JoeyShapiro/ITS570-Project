import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, SAGEConv, Linear, GATConv, GINEConv

class GNN(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GATConv(dataset.num_node_features, dataset.num_edge_features)
        self.conv2 = GATConv(dataset.num_edge_features, 16)#int(dataset.num_node_features/4))
        # self.conv3 = GATConv(int(dataset.num_node_features/4), 16)
        self.classifier = Linear(16, dataset.num_classes) # this 21 was causing issues

    def forward(self, data):
        x, edge_index, edge_attr = torch.Tensor(data.x), data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index)#, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)#, edge_attr)
        # x = self.conv3(x, edge_index)# edge_attr)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)
