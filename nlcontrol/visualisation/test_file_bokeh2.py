from bokeh.io import show, output_file, curdoc, save
import bokeh.plotting as plt
from bokeh.models import (Circle, ColumnDataSource, Ellipse, HoverTool, MultiLine, Oval, PanTool, Plot, Range1d, Rect, Text)

import uuid
import webbrowser

font_size_in_pixels = 15
window_height = 700

nodes = {
    'system': {
        1: {
            'id': uuid.uuid4(),
            'label': 'system', 
            'pos': (2, 0.5),
            'direction': 'right'
            }, 
        2: {
            'id': uuid.uuid4(),
            'label': 'controller',
            'pos': (2, 0),
            'direction': 'left'
            }
        },
    'summation': {
        1: {
            'id': uuid.uuid4(),
            'label': '-1',
            'pos': (1, 0.5)
        }
    }
    }


# Create Bokeh figure
plot = Plot(plot_width=window_height, plot_height=window_height, x_range=Range1d(0.5, 2.5), y_range=Range1d(-0.5, 1.5))

node_hover_tool = HoverTool(tooltips=[("(x,y)", "($x, $y)"),])
plot.add_tools(PanTool(), node_hover_tool)

# Conditions system nodes
nodes_system = nodes['system']
x = []
y = []
text = []
width = []
height = []
in_coords = []
out_coords = []

# Add system text
for node in nodes_system:
    cs = nodes_system[node]
    x.append(cs['pos'][0])
    y.append(cs['pos'][1])
    text.append(cs['label'])
    label_length = len(cs['label'])
    width.append(0.03 * (label_length + 2))
    height.append(0.1)
    left_coord = (x[-1] - width[-1] / 2, y[-1])
    right_coord = (x[-1] + width[-1] / 2, y[-1])
    if cs['direction'] == 'right':
        cs['in_pos'] = left_coord
        cs['out_pos'] = right_coord
    else:
        cs['in_pos'] = right_coord
        cs['out_pos'] = left_coord

source_systems = ColumnDataSource(dict(x=x, y=y, text=text, width=width, height=height))

# Condition sum nodes
nodes_system = nodes['summation']
x = []
y = []
text = []
diameter = []
radius = []
in_coords = []
out_coords = []

# Add system text
for node in nodes_system:
    cs = nodes_system[node]
    x.append(cs['pos'][0])
    y.append(cs['pos'][1])
    text.append(cs['label'])
    label_length = len(cs['label'])
    diameter.append(0.03 * (label_length + 2))
    radius.append(diameter[-1] / 2)
    in_coord = (x[-1], y[-1] - diameter[-1] / 2)
    out_coord = (x[-1] + diameter[-1] / 2, y[-1])
    cs['in_pos'] = in_coord
    cs['out_pos'] = out_coord

source_sum = ColumnDataSource(dict(x=x, y=y, text=text, diameter=diameter, radius=radius))

# Create glyphs
glyph_system_box = Rect(x="x", y="y", width="width", height="height", fill_color="#cab2d6")
glyph_system_text = Text(x="x", y="y", text="text", text_font_size="{}px".format(font_size_in_pixels), text_color="#000000", text_baseline="middle", text_align="center")
glyph_sum_box = Ellipse(x="x", y="y", width="diameter", height="diameter", fill_color="#cab2d6")
glyph_sum_text = Text(x="x", y="y", text="text", text_font_size="{}px".format(font_size_in_pixels), text_color="#000000", text_baseline="middle", text_align="center")

# Add glyphs to plot
plot.add_glyph(source_systems, glyph_system_box)
plot.add_glyph(source_sum, glyph_sum_box)
plot.add_glyph(source_systems, glyph_system_text)
plot.add_glyph(source_sum, glyph_sum_text)

curdoc().add_root(plot)
output_file("interactive_graphs2.html", 'Block scheme')
show(plot, browser='windows-default', new='window')