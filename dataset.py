import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from graph import build_hetero_graph
from torch_geometric.utils.convert import from_networkx


class Networks(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.name = name # TODO deal with me (parent folder)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []#['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        # download_url(url, self.raw_dir)
        pass

    def process(self):
        g, connections = build_hetero_graph('test.pcap') # TODO save and dont do this everytime
        pyg = from_networkx(g)

         # Split the data 
        train_ratio = 0.2
        num_nodes = pyg.x.shape[0]
        num_train = int(num_nodes * train_ratio)
        idx = [i for i in range(num_nodes)]

        np.random.shuffle(idx)
        pyg.train_mask = torch.full_like(pyg.y, False, dtype=bool) # type: ignore
        pyg.train_mask[idx[:num_train]] = True
        pyg.test_mask = torch.full_like(pyg.y, False, dtype=bool) # type: ignore
        pyg.test_mask[idx[num_train:]] = True

        # Read data into huge `Data` list.
        x = torch.eye(pyg.y.size(0), dtype=torch.float) # TODO this one line was causing so many errors dim error [-1 0]
        data = Data(x=x, edge_index=pyg.edge_index, y=pyg.y, train_mask=pyg.train_mask)
        print('pyg:',data)
        # exit(1)
        data_list = [data]

        # if i do something in one of these stpes, print something
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('finally')
        data, slices = self.collate(data_list) # type: ignore

        # self.data, self.slices = data, slices
        torch.save((data, slices), self.processed_paths[0]) # this is how it saves it for use later