import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from graph import build_hetero_graph, graph_from_pcap
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm


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

    def process(self): # this is kinda fun. do something like this later
        data_list = []
        data_base = '/Volumes/T7 Touch/ITS472/project 2/opt/Malware-Project/BigDataset/IoTScenarios/'
        # cant use self.raw_file_names because i need bad ip too
        pcaps = {
            "test.pcap": [],
            # data_base + "CTU-Honeypot-Capture-4-1/2018-10-25-14-06-32-192.168.1.132.pcap": ['192.168.1.132'],
            # data_base + "CTU-Honeypot-Capture-7-1/Somfy-01/2019-07-03-15-15-47-first_start_somfy_gateway.pcap": [],
            # data_base + "CTU-IoT-Malware-Capture-1-1/2018-05-09-192.168.100.103.pcap": [ '192.168.100.103' ]
        }

        pbar = tqdm(pcaps, desc='processing graphs')
        for pcap in pbar:
            pbar.display(f'{pcap} with malicious nodes {pcaps[pcap]}') # not sure if this is done after it finishes
            # pbar.refresh()

            g, connections = graph_from_pcap(pcap, pcaps[pcap])
            pyg = from_networkx(g)

            # Split the data 
            train_ratio = 0.2
            num_nodes = len(pyg.x)#pyg.x.shape[0]
            num_train = int(num_nodes * train_ratio)
            idx = [i for i in range(num_nodes)]

            # create a 3d tensor with padding to make it a perfect rectangle
            length = max(map(len, pyg.z))
            z = torch.from_numpy(np.array([ np.array(z+[[-1, -1, -1]]*(length-len(z)), dtype=np.float32) for z in pyg.z ]))

            # hwew ia a group ondes. these ones are bad
            # bsed on the connections they made, find the ones all the bad ones share
            # based on the bad nodes, what edges are bad and why
            # now mark all links good or bad based on the ones they share
            np.random.shuffle(idx)
            y = torch.Tensor(pyg.y)
            pyg.train_mask = torch.full_like(y, False, dtype=bool) # type: ignore
            pyg.train_mask[idx[:num_train]] = True
            pyg.test_mask = torch.full_like(y, False, dtype=bool) # type: ignore
            pyg.test_mask[idx[num_train:]] = True

            x = torch.eye(y.size(0), dtype=torch.float) # this is node features, so they can be compared; edge attr is different
            # length = max(map(len, pyg.x))
            # x = torch.from_numpy(np.array([ np.array(z+[[-1, -1, -1]]*(length-len(z)), dtype=np.float32) for z in pyg.x ]))
            data = Data(x=x, edge_index=pyg.edge_index, y=y, edge_attr=z, train_mask=pyg.train_mask, test_mask=pyg.test_mask)

            # Read data into huge `Data` list.
            data_list.append(data)

        # if i do something in one of these stpes, print something
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list) # type: ignore

        # self.data, self.slices = data, slices
        torch.save((data, slices), self.processed_paths[0]) # this is how it saves it for use later
