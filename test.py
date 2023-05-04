from gnn import GNN
from dataset import Networks
import torch

dataset = Networks(root='/tmp/Networks', name='test')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(dataset).to(device)
data = dataset[0].to(device) # type: ignore
model.load_state_dict(torch.load("model.pt"))
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')