from bokeh.io import show, output_file, curdoc, save
from bokeh.models import (Arrow, ColumnDataSource, Ellipse, HoverTool, MultiLine, NormalHead, PanTool, Plot, Range1d, Rect, Text)

import uuid

font_size_in_pixels = 15
window_height = 700

number_of_blocks = 3
id_list = [uuid.uuid4() for i in range(number_of_blocks)]

nodes = {
    id_list[0]: {
        'type': 'system',
        'label': 'system', 
        'pos': (2, 0.5),
        'in_direction': 'right',
        'out_direction': 'right',
        'connect_to': [id_list[1]],
        'class_name': 'EulerLagrange',
        'states': 'q_1, dq_1, q_2, dq_2',
        'output': 'q_1, dq_1'
        }, 
    id_list[1]: {
        'type': 'system',
        'label': 'controller',
        'pos': (2, 0),
        'in_direction': 'left',
        'out_direction': 'left',
        'connect_to': [id_list[2]],
        'class_name': 'EulerLagrangeController',
        'states': 'p_1, p_2',
        'output': 'p_2'
        },
    id_list[2]: {
        'type': 'summation',
        'label': '-1',
        'pos': (1, 0.5),
        'in_direction': 'up',
        'out_direction': 'right',
        'connect_to': [id_list[0]],
        'output' : '-p_2'
    }
}


# Create Bokeh figure
plot = Plot(plot_width=window_height, plot_height=window_height, x_range=Range1d(0.5, 2.5), y_range=Range1d(-0.5, 1.5))

# Conditions system nodes
x = []
y = []
text = []
width = []
height = []
in_coords = []
out_coords = []
classname = []
states = []
output = []

# Add system text
for node in nodes:
    cs = nodes[node]
    if cs['type'] == 'system':
        x.append(cs['pos'][0])
        y.append(cs['pos'][1])
        text.append(cs['label'])
        label_length = len(cs['label'])
        width.append(0.03 * (label_length + 2))
        height.append(0.1)
        left_coord = (x[-1] - width[-1] / 2, y[-1])
        right_coord = (x[-1] + width[-1] / 2, y[-1])
        if cs['in_direction'] == 'right':
            cs['in_pos'] = left_coord
            cs['out_pos'] = right_coord
        else:
            cs['in_pos'] = right_coord
            cs['out_pos'] = left_coord
        classname.append(cs['class_name'])
        states.append(cs['states'])
        output.append(cs['output'])

source_systems = ColumnDataSource(dict(x=x, y=y, text=text, width=width, height=height, data1=classname, data2=states, data3=output))

# Condition sum nodes
x = []
y = []
text = []
diameter = []
radius = []
in_coords = []
out_coords = []

# Add system text
for node in nodes:
    cs = nodes[node]
    if cs['type'] == 'summation':
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

# Generate connection polynomials
x_polynomials = []
y_polynomials = []
output = []

for node in nodes:
    cs = nodes[node]
    polyn_x_coords = [] 
    polyn_y_coords = []
    if len(cs['connect_to']) == 0:
        # Begin coordinate
        polyn_x_coords.append(cs['out_pos'][0])
        polyn_y_coords.append(cs['out_pos'][1])
        # End coordinate
        sign = 1 if cs['out_direction'] == 'right' else -1
        half_width = sign * 0.5 * abs(cs['out_pos'][0] - cs['in_pos'][0])
        polyn_x_coords.append(cs['out_pos'][0] + half_width)
        polyn_y_coords.append(cs['out_pos'][1])
        x_polynomials.append(polyn_x_coords)
        y_polynomials.append(polyn_y_coords)
        output.append(cs['output'])
    else:
        for connection in cs['connect_to']:
            end_block = nodes[connection]
            # Begin coordinate
            polyn_x_coords.append(cs['out_pos'][0])
            polyn_y_coords.append(cs['out_pos'][1])
            # Neighbor coordinate start block
            sign = 1 if cs['out_direction'] == 'right' else -1
            width = abs(cs['out_pos'][0] - cs['in_pos'][0])
            polyn_x_coords.append(cs['out_pos'][0] + sign * 1.2 * width)
            polyn_y_coords.append(cs['out_pos'][1])

            # Neigbor coordinate end block
            if end_block['in_direction'] == 'up':
                polyn_x_coords.append(end_block['in_pos'][0])
                polyn_y_coords.append(polyn_y_coords[-1])
            else:
                polyn_x_coords.append(polyn_x_coords[-1])
                polyn_y_coords.append(end_block['in_pos'][1])
            # End block coordinates
            polyn_x_coords.append(end_block['in_pos'][0])
            polyn_y_coords.append(end_block['in_pos'][1])
            x_polynomials.append(polyn_x_coords)
            y_polynomials.append(polyn_y_coords)
            output.append(cs['output'])
            polyn_x_coords = []
            polyn_y_coords = []

print(nodes)
print(x_polynomials)
print(y_polynomials)

# Create glyphs
glyph_system_box = Rect(x="x", y="y", width="width", height="height", fill_color="#cab2d6")
glyph_system_text = Text(x="x", y="y", text="text", text_font_size="{}px".format(font_size_in_pixels), text_color="#000000", text_baseline="middle", text_align="center")
glyph_sum_box = Ellipse(x="x", y="y", width="diameter", height="diameter", fill_color="#cab2d6")
glyph_sum_text = Text(x="x", y="y", text="text", text_font_size="{}px".format(font_size_in_pixels), text_color="#000000", text_baseline="middle", text_align="center")

# Add block glyphs to plot
glyph_system_box_renderer = plot.add_glyph(source_systems, glyph_system_box)
plot.add_glyph(source_sum, glyph_sum_box)
glyph_system_text_renderer = plot.add_glyph(source_systems, glyph_system_text)
plot.add_glyph(source_sum, glyph_sum_text)

# Add arrows
for x_poly, y_poly in zip(x_polynomials, y_polynomials):
    glyph_arrow = Arrow(end=NormalHead(size=15, fill_color="#000000"),
                           x_start=x_poly[-2],
                           y_start=y_poly[-2],
                           x_end=x_poly[-1],
                           y_end=y_poly[-1],
                           line_width=0,
                           line_color = "#000000")
    plot.add_layout(glyph_arrow)

# Add lines
source_lines = ColumnDataSource(dict(xs=x_polynomials, ys=y_polynomials, output=output))
glyph_lines = MultiLine(xs="xs", ys="ys", line_color="#000000", line_width=3)
glyph_lines_renderer = plot.add_glyph(source_lines, glyph_lines)


# node_hover_tool = HoverTool(tooltips=[("(x,y)", "($x, $y)"),])
node_hover_tool = HoverTool(line_policy="nearest", tooltips="""
<div>
    <div>
        <b>Class Name:</b>
        <i>@data1 </i>
    </div>
    <div>
        <b>States:</b>
        <i>@data2</i>
    </div>
    <div>
        <b>Outputs:</b>
        <i>@data3</i>
    </div>
</div>
    """,
    renderers=[glyph_system_box_renderer, glyph_system_text_renderer])

# Should work from Bokeh 2.3.x
line_hover_tool = HoverTool(line_policy="nearest", tooltips="""
<div>
    <div>
        <b>Output:</b>
        <i>@output </i>
    </div>
</div>
    """,
    renderers=[glyph_lines_renderer]
)
plot.add_tools(PanTool(), node_hover_tool, line_hover_tool)


curdoc().add_root(plot)
output_file("closed_loop.html", 'Closed Loop Block scheme')
show(plot, browser='windows-default', new='window')