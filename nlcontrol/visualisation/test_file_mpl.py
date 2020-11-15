import networkx as nx
import matplotlib.pyplot as plt

from helper_fcts import SystemBlock

G = nx.DiGraph()
# Define nodes
G.add_node('1a', type='conn')
G.add_node(1, type='sys', label='system')
G.add_node('1b', type='conn')
G.add_node('2a', type='conn')
G.add_node(2, type='sys', label='controller')
G.add_node('2b', type='conn')
G.add_node(3, type='sum', label='-1')

# print(list(G.nodes(data=True)))

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

# print(labels_sys)
# print(list(G.nodes(data=True)))

# Drawing
pad = 1.1
bbox_sys = dict({'edgecolor': 'black', 'facecolor': 'white', 'alpha': 0.5, 'boxstyle': 'square,pad={}'.format(pad)})
bbox_sum = dict({'edgecolor': 'black', 'facecolor': 'white', 'boxstyle': 'circle,pad={}'.format(pad)})
pos = {'1a': (1.8, 0.5), 1: (2, 0.5), '1b': (2.2, 0.5), '2a': (2.2, 0), 2: (2, 0), '2b': (1.8, 0), 3: (1, 0.5)}

# Create figure with matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, frame_on=False, aspect='auto')

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
pix_per_unit = ax.transData.transform([(0,1)])-ax.transData.transform((0,0))[1]
# Get dimensions of axes in pixels
y1, y2 = ax.get_window_extent().get_points()[:, 1]
x1, x2 = ax.get_window_extent().get_points()[:, 0]
# Get limits in axis coords
ymin, ymax = ax.get_ylim()
xmin, xmax = ax.get_xlim()
# Get unit scale
yscale = (y2 - y1)/(ymax - ymin)
xscale = (x2 - x1)/(xmax - xmin)

renderer = fig.canvas.get_renderer()
# pixel_per_points = renderer.points_to_pixels(14 * 1.1)
# unit_per_points = pixel_per_points/yscale)
for el in ax.get_children():
    if type(el) == type(plt.Text()) and el._bbox_patch is not None:
        pixel_per_points = renderer.points_to_pixels(pad * el.get_fontsize())
        pad_in_unit_per_points_y = pixel_per_points/yscale
        pad_in_unit_per_points_x = pixel_per_points/xscale
        # print(pad_in_unit_per_points_x, ' : ', pad_in_unit_per_points_y)
        
        # Find the bounded box
        bb = el.get_window_extent(renderer)
        bbox_without_pad = bb.transformed(transf)
        # print(bbox_without_pad)
        # print(bbox_without_pad.x0 - pad_in_unit_per_points_x, " - ", bbox_without_pad.y0 - pad_in_unit_per_points_y)
        sys = SystemBlock(bbox_without_pad, pad_in_unit_per_points_x, pad_in_unit_per_points_y)

nx.draw_networkx_edges(G,
    ax=ax,
    pos=pos)

plt.show()

