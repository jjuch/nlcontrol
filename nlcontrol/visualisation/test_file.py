import networkx as nx

from bokeh.io import show
from bokeh.plotting import figure

import matplotlib.pyplot as plt

G = nx.DiGraph()
# Define nodes
G.add_node('1a', type='conn')
G.add_node(1, type='sys', label='system')
G.add_node('1b', type='conn')
G.add_node('2a', type='conn')
G.add_node(2, type='sys', label='controller')
G.add_node('2b', type='conn')
G.add_node(3, type='sum', label='-1')

print(list(G.nodes(data=True)))

# Define edges
G.add_edges_from([('1a', 1), (1, '1b'), ('1b', '2a'), ('2a', 2), (2, '2b'), ('2b', 3), (3, '1a')])

# extract nodes with specific setting of the attribute 'type'
system_nodes = [n for (n, ty) in \
    nx.get_node_attributes(G, 'type').items() if ty == 'sys']
sum_nodes = [n for (n, ty) in \
    nx.get_node_attributes(G, 'type').items() if ty == 'sum']
conn_nodes = [n for (n, ty) in \
    nx.get_node_attributes(G, 'type').items() if ty == 'conn']
remainder_nodes = list(set(G.nodes()) - set(system_nodes) - set(sum_nodes) - set(conn_nodes))

labels_sys = {n: G.nodes[n]['label'] for n in system_nodes}
labels_sum = {n: G.nodes[n]['label'] for n in sum_nodes}

print(labels_sys)
# print(list(G.nodes(data=True)))

# Drawing
bbox_sys = dict({'edgecolor': 'black', 'facecolor': 'white', 'alpha': 0.5, 'boxstyle': 'square,pad=1.1'})
bbox_sum = dict({'edgecolor': 'black', 'facecolor': 'white', 'boxstyle': 'circle,pad=1.1'})
pos = {'1a': (1.8, 0.5), 1: (2, 0.5), '1b': (2.2, 0.5), '2a': (2.2, 0), 2: (2, 0), '2b': (1.8, 0), 3: (1, 0.5)}

# nx.draw_networkx(G, 
#     pos=pos, 
#     nodelist=system_nodes,
#     node_color='white', 
#     with_labels=True, 
#     bbox=bbox_sys, 
#     labels=labels_sys,
#     arrows=False)
# nx.draw_networkx(G,
#     pos=pos, 
#     nodelist=sum_nodes, 
#     with_labels=True, 
#     bbox=bbox_sum, 
#     labels=labels_sum)
# nx.draw_networkx(G,
#     pos=pos, 
#     nodelist=conn_nodes, 
#     with_labels=False,
#     node_size=0,
#     arrows=False)
# plt.show()

# Create figure
fig = plt.figure()
ax = fig.add_subplot(111)

# Draw nodes
nx.draw_networkx_nodes(G,
    ax=ax,
    pos=pos,
    nodelist=system_nodes,
    node_color='red',
    node_shape='s')

nx.draw_networkx_nodes(G,
    ax=ax,
    pos=pos,
    nodelist=sum_nodes,
    node_color='red')

nx.draw_networkx_nodes(G,
    ax=ax,
    pos=pos,
    nodelist=conn_nodes,
    node_color='red',
    node_size=0.1)

# Draw labels
nx.draw_networkx_labels(G,
    ax=ax,
    pos=pos,
    labels=labels_sys,
    font_size=14,
    bbox=bbox_sys)

nx.draw_networkx_labels(G,
    ax=ax,
    pos=pos,
    labels=labels_sum,
    bbox=bbox_sum)

transf = ax.transData.inverted()
# Get dimensions of y-axis in pixels
y1, y2 = ax.get_window_extent().get_points()[:, 1]
# Get limits in axis coords
ymin, ymax = ax.get_ylim()
# Get unit scale
yscale = (y2-y1)/(ymax-ymin)
padd_in_pix = 1.1*yscale
print(padd_in_pix)
for el in ax.get_children():
    if type(el) == type(plt.Text()):
        bb = el.get_window_extent(fig.canvas.get_renderer())
        print(bb)
        print(el, ': ', bb.transformed(transf))
        # TODO: find padding distance in units per point times the padding.

nx.draw_networkx_edges(G,
    ax=ax,
    pos=pos)

plt.show()