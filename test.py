from gnn import GNN
from dataset import Networks
import torch

dataset = Networks(root='/tmp/Networks', name='test')

model = GNN(dataset)
model.load_state_dict(torch.load("model.pt"))
model.eval()