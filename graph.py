import networkx as nx
import scapy.all as sa
from scapy.layers.inet import IP, UDP, TCP # this is its
import numpy as np

# fix weird scapy bug

def build_hetero_graph(pcap, display=False):
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

            if packet[IP].src not in nodes:
                nodes.append(packet[IP].src)
            if packet[IP].dst not in nodes:
                nodes.append(packet[IP].dst)

            if packet[2].name not in transports:
                transports.append(packet[2].name)
            
            if packet.haslayer(UDP):
                dport = packet[UDP].dport
                sport = packet[UDP].sport
                print(f'UDP {packet[IP].src}:{packet[UDP].sport} -> {packet[IP].dst}:{packet[UDP].dport}')
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
    g.add_nodes_from([ (i, { "y": i, "x": 1.0 }) for i in range(len(nodes)) ]) # convert to index list; they will match up
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