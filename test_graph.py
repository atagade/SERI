import networkx as nx
import matplotlib.pyplot as plt
G = nx.read_graphml('taxi_graph.graphml')
nx.draw(G, with_labels=True)
plt.show()
