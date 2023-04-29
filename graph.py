import networkx as nx
import scapy.all as sa
from scapy.layers.inet import IP, UDP, TCP # this is its

# fix weird scapy bug

def build_hetero_graph(pcap, display=False):
    g = nx.Graph()

    connections = {}
    nodes = []
    pcap_flow = sa.rdpcap(pcap)
    sessions = pcap_flow.sessions()
    for session in sessions:
        for packet in sessions[session]:
            dport = None
            sport = None
            if packet[IP].src not in nodes:
                nodes.append(packet[IP].src)
            if packet[IP].dst not in nodes:
                nodes.append(packet[IP].dst)
            
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
                        { 'protocol': packet[2].name, 'sport': sport, 'dport': dport }
                    ]
                } # add all connections, seems better for now
            else:
                connections[(nodes.index(packet[IP].src), nodes.index(packet[IP].dst))]['packets'].append(
                    { 'protocol': packet[2].name, 'sport': sport, 'dport': dport }
                )

#     g.add_nodes_from([
#       (1, {'y': 1, 'x': 0.5}),
#       (2, {'y': 2, 'x': 0.2}),
#       (3, {'y': 3, 'x': 0.3}),
#       (4, {'y': 4, 'x': 0.1}),
#       (5, {'y': 5, 'x': 0.2}),
# ])
#     g.add_edges_from([
#                   (1, 2), (1, 4), (1, 5),
#                   (2, 3), (2, 4),
#                   (3, 2), (3, 5),
#                   (4, 1), (4, 2),
#                   (5, 1), (5, 3)
# ])
    
    # it can just find x and y when converting
    g.add_nodes_from([ (i, { "y": i, "x": 1.0 }) for i in range(len(nodes)) ]) # convert to index list
    g.add_edges_from(connections.keys())
    str_conns = {} # whatever it works, so just keep it as iss
    for conn in connections:
        # print(str(connections[conn].values()))
        str_conns[conn] = { "z": 1.0 }#str(connections[conn].values())}
    # str_conns = [ str(conn['packets']) for conn in connections.values() ]
    nx.set_edge_attributes(g, str_conns) # type: ignore

    if display:
        pos = nx.kamada_kawai_layout(g) # type: ignore
        nx.draw(g, pos) # type: ignore
        import matplotlib.pyplot as plt
        plt.show()

    return g, connections