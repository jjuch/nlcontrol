import networkx as nx
import pandas as pd

from bokeh.io import show, output_file
import bokeh.plotting as boplt
from bokeh.models import (Circle, ColumnDataSource, EdgesAndLinkedNodes,
                        HoverTool, MultiLine, NodesAndLinkedEdges, PanTool, Plot, Range1d, Rect, Text)
from bokeh.palettes import Spectral4

from helper_fcts import SystemBlock

G = nx.DiGraph()

# Define nodes
G.add_node(10001, type='conn')
G.add_node(1, type='sys', label='system')
G.add_node(20001, type='conn')
G.add_node(10002, type='conn')
G.add_node(2, type='sys', label='controller')
G.add_node(20002, type='conn')
G.add_node(3, type='sum', label='-1')

# Define edges
G.add_edges_from([(10001, 1), (1, 20001), (20001, 10002), (10002, 2), (2, 20002), (20002, 3), (3, 10001)])

# Positions
pos = {10001: (1.8, 0.5), 1: (2, 0.5), 20001: (2.2, 0.5), 10002: (2.2, 0), 2: (2, 0), 20002: (1.8, 0), 3: (1, 0.5)}

# extract nodes with specific setting of the attribute 'type'
system_nodes = [n for (n, ty) in \
    nx.get_node_attributes(G, 'type').items() if ty == 'sys']
sum_nodes = [n for (n, ty) in \
    nx.get_node_attributes(G, 'type').items() if ty == 'sum']
conn_nodes = [n for (n, ty) in \
    nx.get_node_attributes(G, 'type').items() if ty == 'conn']
remainder_nodes = list(set(G.nodes()) - set(system_nodes) - set(sum_nodes) - set(conn_nodes))

labels_sys = {n: {'label': G.nodes[n]['label'], 'x': pos[n][0], 'y': pos[n][1], 'width': 0.3, 'height': 0.1} for n in system_nodes}
labels_sum = {n: {'label': G.nodes[n]['label'], 'x': pos[n][0], 'y': pos[n][1], 'width': 0.3, 'height': 0.1} for n in sum_nodes}

labels = dict(labels_sys)
labels.update(labels_sum)

# Bokeh sources
sources_labels = ColumnDataSource(pd.DataFrame.from_dict(labels, orient='index'))
sources_boxes = ColumnDataSource(pd.DataFrame.from_dict(labels, orient='index'))

#set node attributes
# node_color = {'A': Viridis[10][0], 'B': Viridis[10][9], 'a': Viridis[10][4],'b': Viridis[10][4]}
# node_size = {'A':50, 'B':40,'a':10,'b':10}
# node_initial_pos = {'A': (-0.5,0), 'B': (0.5,0),'a': (0,0.25),'b': (0,-0.25)}
# nx.set_node_attributes(G, 'node_color', node_color)
# nx.set_node_attributes(G, 'node_size', node_size)
# nx.set_node_attributes(G, 'node_initial_pos', node_initial_pos)

#source with node color, size and initial pos (perhaps )
source = ColumnDataSource(pd.DataFrame.from_dict({k:v for k,v in G.nodes(data=True)}, orient='index'))



# Create a figure in Bokeh
plot = Plot(plot_width=800, plot_height=700,
            x_range=Range1d(0.5, 2.5), y_range=Range1d(-0.5, 1))

node_hover_tool = HoverTool(tooltips=[("(x,y)", "($x, $y)"),])
plot.add_tools(PanTool(), node_hover_tool)

graph_renderer_labels = boplt.from_networkx(G, layout_function=pos, scale=1, center=(0,0))
graph_renderer_boxes = boplt.from_networkx(G, layout_function=pos, scale=1, center=(0,0))


# Make labels
graph_renderer_labels.node_renderer.data_source = sources_labels
graph_renderer_labels.node_renderer.glyph = Text(text='label', text_font_size='15px', text_color="#000000", text_baseline="middle", text_align="center")
# graph_renderer.node_renderer.hover_glyph = Text(text='label',text_font_size='15px', text_color=Spectral4[1])

# Make boxes
graph_renderer_boxes.node_renderer.data_source = sources_boxes
graph_renderer_boxes.node_renderer.glyph = Rect(x='x', y='y', width='width', height="height", fill_color='#FFFFFF')
# graph_renderer.node_renderer.hover_glyph = Text(text='label',text_font_size='15px', text_color=Spectral4[1])

# Make edges
graph_renderer_labels.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)
graph_renderer_labels.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
graph_renderer_labels.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

# graph_renderer.selection_policy = NodesAndLinkedEdges()
# graph_renderer.inspection_policy = EdgesAndLinkedNodes()

plot.renderers.extend([graph_renderer_boxes, graph_renderer_labels])
# plot.renderers.append(graph_renderer_labels)
print(plot.renderers)

output_file("interactive_graphs.html")
show(plot)