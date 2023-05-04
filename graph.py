import networkx as nx
import scapy.all as sa
from scapy.layers.inet import IP, UDP, TCP # this is its
import numpy as np
from tqdm import tqdm
import os

# could parse out header info and make it smaller and easier to work with
# only what i need but not as fun and that more work. pcap has whole packet, 
# dont need all that. but who knows what i need
def graph_from_pcap(pcap, bad, display=False):
    g = nx.Graph()

    # not 100% accuracte, but close enough
    # TODO maybe remove some percents
    pbar = tqdm(desc='parsing packets', total=os.path.getsize(pcap)) # tqdm doesnt need to be in a for loop

    connections = {}
    nodes = []
    transports = [] # need index, so better then dict

    def parse_pkt(packet):
        dport = -1
        sport = -1

        if not packet.haslayer(IP):
            pbar.update(len(packet))
            return

        if packet[IP].src not in nodes:
            nodes.append(packet[IP].src)
        if packet[IP].dst not in nodes:
            nodes.append(packet[IP].dst)

        if packet[2].name not in transports:
            transports.append(packet[2].name)
        
        if packet.haslayer(UDP):
            dport = packet[UDP].dport
            sport = packet[UDP].sport
            # pbar.display(f'UDP {packet[IP].src}:{packet[UDP].sport} -> {packet[IP].dst}:{packet[UDP].dport}') # tqdm this
        elif packet.haslayer(sa.TCP_SERVICES):
            dport = packet[TCP].dport
            sport = packet[TCP].sport
            # pbar.display(f'TCP {packet[IP].src}:{packet[TCP].sport} -> {packet[IP].dst}:{packet[TCP].dport}')

        if (nodes.index(packet[IP].src),nodes.index(packet[IP].dst)) not in connections:
            connections[(nodes.index(packet[IP].src), nodes.index(packet[IP].dst))] = {
                'packets': [
                    { 'protocol': transports.index(packet[2].name), 'sport': sport, 'dport': dport }
                ]
            } # add all connections, seems better for now
        else:
            connections[(nodes.index(packet[IP].src), nodes.index(packet[IP].dst))]['packets'].append(
                { 'protocol': transports.index(packet[2].name), 'sport': sport, 'dport': dport }
            )
        # nonlocal total # -_-
        # total += len(packet)
        pbar.update(len(packet))

    # TODO use smaller files; batch load
    sa.sniff(offline=pcap, prn=parse_pkt, store=0)
    
    g.add_nodes_from([ (i, { "y": 1 if ip in bad else 0, "x": 1.0 }) for i, ip in enumerate(nodes) ]) # convert to index list; they will match up
    g.add_edges_from(connections.keys())
    str_conns = {} # whatever it works, so just keep it as iss
    for conn in tqdm(connections, desc='aggregating links'):
        # print(str(connections[conn].values()))
        data = []
        for pkt in connections[conn]['packets']:
            data.append([ pkt['protocol'], pkt['sport'], pkt['dport'] ])
        str_conns[conn] = { "z": data }
    # str_conns = [ str(conn['packets']) for conn in connections.values() ]
    nx.set_edge_attributes(g, str_conns) # type: ignore

    # edge attributes are added to the src ip node
    # for conn in connections:
    #     g.nodes[conn[0]]['x'] = str_conns[conn]['z'][0][0]

    if display:
        pos = nx.kamada_kawai_layout(g) # type: ignore
        nx.draw(g, pos) # type: ignore
        import matplotlib.pyplot as plt
        plt.show()

    return g, connections

def build_hetero_graph(pcap, bad, display=False):
    g = nx.Graph()

    connections = {}
    nodes = []
    transports = [] # need index, so better then dict
    pcap_flow = sa.rdpcap(pcap)
    sessions = pcap_flow.sessions()
    for session in sessions:
        for packet in sessions[session]:
            dport = -1 # TODO what to do if no data
            sport = -1

            if not packet.haslayer(IP):
                continue

            if packet[IP].src not in nodes:
                nodes.append(packet[IP].src)
            if packet[IP].dst not in nodes:
                nodes.append(packet[IP].dst)

            if packet[2].name not in transports:
                transports.append(packet[2].name)
            
            if packet.haslayer(UDP):
                dport = packet[UDP].dport
                sport = packet[UDP].sport
                print(f'UDP {packet[IP].src}:{packet[UDP].sport} -> {packet[IP].dst}:{packet[UDP].dport}') # tqdm this
            elif packet.haslayer(sa.TCP_SERVICES):
                dport = packet[TCP].dport
                sport = packet[TCP].sport
                print(f'TCP {packet[IP].src}:{packet[TCP].sport} -> {packet[IP].dst}:{packet[TCP].dport}')

            if (nodes.index(packet[IP].src),nodes.index(packet[IP].dst)) not in connections:
                connections[(nodes.index(packet[IP].src), nodes.index(packet[IP].dst))] = {
                    'packets': [
                        { 'protocol': transports.index(packet[2].name), 'sport': sport, 'dport': dport }
                    ]
                } # add all connections, seems better for now
            else:
                connections[(nodes.index(packet[IP].src), nodes.index(packet[IP].dst))]['packets'].append(
                    { 'protocol': transports.index(packet[2].name), 'sport': sport, 'dport': dport }
                )
    
    # it can just find x and y when converting
    # y is mask, 0 = good
    # this as i wasnt able to learn anything
    # everything was different
    # other datasets have less y's
    # those worked
    # now works
    # was commented out part, gnn not perfect, y was every value, nothing to learn
    # just need to give real data and test different models
    # y is every value so it couldnt find anything differnt
    g.add_nodes_from([ (i, { "y": 1 if ip in bad else 0, "x": 1.0 }) for i, ip in enumerate(nodes) ]) # convert to index list; they will match up
    g.add_edges_from(connections.keys())
    str_conns = {} # whatever it works, so just keep it as iss
    for conn in connections:
        # print(str(connections[conn].values()))
        data = []
        for pkt in connections[conn]['packets']:
            data.append([ pkt['protocol'], pkt['sport'], pkt['dport'] ])
        str_conns[conn] = { "z": data }
    # str_conns = [ str(conn['packets']) for conn in connections.values() ]
    nx.set_edge_attributes(g, str_conns) # type: ignore

    # edge attributes are added to the src ip node
    # for conn in connections:
    #     g.nodes[conn[0]]['x'] = str_conns[conn]['z'][0][0]

    if display:
        pos = nx.kamada_kawai_layout(g) # type: ignore
        nx.draw(g, pos) # type: ignore
        import matplotlib.pyplot as plt
        plt.show()

    return g, connections
