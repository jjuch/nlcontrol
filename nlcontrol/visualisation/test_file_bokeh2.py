from bokeh.io import show, output_file, curdoc
import bokeh.plotting as plt
from bokeh.models import (Circle, ColumnDataSource, HoverTool, MultiLine, PanTool, Plot, Range1d, Rect, Text)

import uuid

font_size_in_pixels = 15

nodes = {
    'system': {
        1: {
            'id': uuid.uuid4(),
            'label': 'system', 
            'pos': (2, 0.5),
            'in_pos': (1.8, 0.5),
            'out_pos': (2.2, 0.5) 
            }, 
        2: {
            'id': uuid.uuid4(),
            'label': 'controller',
            'pos': (2, 0),
            'in_pos': (2.2, 0),
            'out_pos': (1.8, 0)
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

# nodes_system = nodes['system']
# for node in nodes_system:
#     print('node = ', nodes_system[node])

# Create Bokeh figure
plot = Plot(plot_width=800, plot_height=700, x_range=Range1d(0.5, 2.5), y_range=Range1d(-0.5, 1))

node_hover_tool = HoverTool(tooltips=[("(x,y)", "($x, $y)"),])
plot.add_tools(PanTool(), node_hover_tool)

nodes_system = nodes['system']
x = []
y = []
text = []
width = []
height = []

# Add system text
for node in nodes_system:
    cs = nodes_system[node]
    x.append(cs['pos'][0])
    y.append(cs['pos'][1])
    text.append(cs['label'])
    label_length = len(cs['label'])
    # width.append('{}px'.format((label_length + 2) * 15))
    # height.append('{}px'.format(font_size_in_pixels + 2))
    width.append(0.3)
    height.append(0.1)

print(width)
print(height)

source = ColumnDataSource(dict(x=x, y=y, text=text, width=width, height=height))

# Add system box
glyph_box = Rect(x="x", y="y", width="width", height="height", fill_color="#cab2d6")
plot.add_glyph(source, glyph_box)

glyph_text = Text(x='x', y='y', text='text', text_font_size='{}px'.format(font_size_in_pixels), text_color="#000000", text_baseline="middle", text_align="center")
plot.add_glyph(source, glyph_text)


curdoc().add_root(plot)
show(plot)