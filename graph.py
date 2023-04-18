import networkx as nx
import scapy.all as sa
from scapy.layers.inet import IP, UDP, TCP # this is its

# fix weird scapy bug

def build_hetero_graph(pcap):
    g = nx.Graph()

    connections = {}
    pcap_flow = sa.rdpcap(pcap)
    sessions = pcap_flow.sessions()
    for session in sessions:
        for packet in sessions[session]:
            dport = None
            sport = None
            if packet.haslayer(UDP):
                dport = packet[UDP].dport
                sport = packet[UDP].sport
                print(f'UDP {packet[IP].src}:{packet[UDP].sport} -> {packet[IP].dst}:{packet[UDP].dport}')
            elif packet.haslayer(sa.TCP_SERVICES):
                dport = packet[TCP].dport
                sport = packet[TCP].sport
                print(f'TCP {packet[IP].src}:{packet[TCP].sport} -> {packet[IP].dst}:{packet[TCP].dport}')
            if (packet[IP].src,packet[IP].dst) not in connections:
                connections[(packet[IP].src,packet[IP].dst)] = {
                    'packets': [
                        { 'protocol': packet[2].name, 'sport': sport, 'dport': dport }
                    ]
                } # add all connections, seems better for now
            else:
                connections[(packet[IP].src,packet[IP].dst)]['packets'].append(
                    { 'protocol': packet[2].name, 'sport': sport, 'dport': dport }
                )

    # g.add_edges_from(connections)
    g.add_edges_from(connections.keys())
    nx.set_edge_attributes(g, connections) # type: ignore
    print(g['10.215.28.18']['10.215.63.255']['packets']) # TODO for later, could add ports to this

    pos = nx.kamada_kawai_layout(g) # type: ignore
    nx.draw(g, pos) # type: ignore
    import matplotlib.pyplot as plt
    plt.show()

    return g