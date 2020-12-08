from nlcontrol.visualisation.file_management import __write_to_browser__
from nlcontrol.visualisation.drawing_tools import draw_line, generate_system_renderer_info, generate_parallel_renderer_info, generate_renderer_sources
from nlcontrol.visualisation.utils import pretty_print_dict

from bokeh.io import show, output_file, curdoc
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.models import (Arrow, ColumnDataSource, Ellipse, HoverTool, MultiLine, NormalHead, PanTool, Plot, Range1d, Rect, Text)

import uuid

FONT_SIZE_IN_PIXELS = 15
x_offset, y_offset = 0, 0

class RendererBase():
    def __init__(self, **kwargs):
        self.plot = None
        self.plot_dict = dict()
        self.renderer_info = self.__init_renderer_info__(**kwargs)

    def __init_renderer_info__(self, block_type="system", **kwargs):
        unique_id = uuid.uuid4().hex
        info = {unique_id : dict()}
        info_id = info[unique_id]
        if 'block_type' in kwargs:
            block_type = kwargs['block_type']
        print(block_type)

        if block_type == "system":
            info_dict = generate_system_renderer_info(self)
        elif block_type == "parallel":
            if 'systems' not in kwargs:
                error_text = "[RendererBase] In the case of a 'parallel' block_type a keyword argument 'systems' should be applied."
                raise AttributeError(error_text)
            info_dict = generate_parallel_renderer_info(self, kwargs['systems'])
        info_id.update(info_dict)
        return info

    
    def set_coordinates(self, current_element=None):
        if current_element is None:
            current_element = self.renderer_info
        for current_id in current_element.keys():
            current_data = current_element[current_id]
            current_data['position'] = current_data['rel_position'](current_data['x_offset'], current_data['y_offset'])
            if 'nodes' in current_data:
                self.set_coordinates(current_element=current_data['nodes'])
        # print(self.renderer_info)


    def create_connections(self):
        x_polynomials = []
        y_polynomials = []
        output = []

        for node in self.renderer_info:
            cs = self.renderer_info[node]
            polyn_x_coords = [] 
            polyn_y_coords = []
            if cs['type'] != 'common':
                width = abs(cs['out_pos'][0] - cs['in_pos'][0])
                if len(cs['connect_from']) == 0:
                    # End coordinate
                    polyn_x_coords.append(cs['in_pos'][0])
                    polyn_y_coords.append(cs['in_pos'][1])
                    # Begin coordinate
                    sign = -1 if cs['in_direction'] == 'right' else 1
                    polyn_x_coords.append(cs['in_pos'][0] + sign * width)
                    polyn_y_coords.append(cs['in_pos'][1])

                    # Add flipped coordinates
                    polyn_x_coords.reverse()
                    polyn_y_coords.reverse()
                    x_polynomials.append(polyn_x_coords)
                    y_polynomials.append(polyn_y_coords)
                    output.append(None)

                    polyn_x_coords = []
                    polyn_y_coords = []

                if len(cs['connect_to']) == 0:
                    # Begin coordinate
                    polyn_x_coords.append(cs['out_pos'][0])
                    polyn_y_coords.append(cs['out_pos'][1])
                    # End coordinate
                    sign = 1 if cs['out_direction'] == 'right' else -1
                    polyn_x_coords.append(cs['out_pos'][0] + sign * width)
                    polyn_y_coords.append(cs['out_pos'][1])
                    x_polynomials.append(polyn_x_coords)
                    y_polynomials.append(polyn_y_coords)
                    output.append(cs['output'])
                else:
                    for connection in cs['connect_to']:
                        end_block = nodes[connection]
                        if type(end_block['in_direction']) == list:
                            if cs['position'][1] > end_block['position'][1]:
                                end_block_in = end_block['in_pos'][0]
                            else:
                                end_block_in = end_block['in_pos'][1]
                        else:
                            end_block_in = end_block['in_pos']
                        # Begin coordinate
                        polyn_x_coords.append(cs['out_pos'][0])
                        polyn_y_coords.append(cs['out_pos'][1])

                        polyn_x_coords.append(end_block_in[0])
                        polyn_y_coords.append(cs['out_pos'][1])

                        # End block coordinates
                        polyn_x_coords.append(end_block_in[0])
                        polyn_y_coords.append(end_block_in[1])
                        x_polynomials.append(polyn_x_coords)
                        y_polynomials.append(polyn_y_coords)
                        output.append(cs['output'])
                        polyn_x_coords = []
                        polyn_y_coords = []
            else:
                for connection in cs['connect_to']:
                    end_block = nodes[connection]
                    end_block_in = end_block['in_pos']
                    # Begin coordinate
                    polyn_x_coords.append(cs['out_pos'][0])
                    polyn_y_coords.append(cs['out_pos'][1])

                    polyn_x_coords.append(polyn_x_coords[-1])
                    polyn_y_coords.append(end_block_in[1])

                    # End block coordinates
                    polyn_x_coords.append(end_block_in[0])
                    polyn_y_coords.append(end_block_in[1])
                    x_polynomials.append(polyn_x_coords)
                    y_polynomials.append(polyn_y_coords)
                    output.append(cs['output'])
                    polyn_x_coords = []
                    polyn_y_coords = []
                

        return x_polynomials, y_polynomials, output


    def show(self, open=True):
        print("Showing the system {} with name '{}'".format(type(self), self.name))
        self.set_coordinates()
        
        # source_systems, source_sum, source_commons = self.create_sources()
        source_systems, source_sum, source_commons = generate_renderer_sources(self.renderer_info)
        # pretty_print_dict(self.renderer_info)
        exit()

        x_polynomials, y_polynomials, output_lines = self.create_connections()

        glyph_system_text = Text(x="x", y="y", text="text", text_font_size="{}px".format(FONT_SIZE_IN_PIXELS), text_color="#000000", text_baseline="middle", text_align="center")
        glyph_system_box = Rect(x="x", y="y", width="width", height="height", fill_color="#cab2d6", fill_alpha=0.4)
        glyph_sum_text = Text(x="x", y="y", text="text", text_font_size="{}px".format(FONT_SIZE_IN_PIXELS), text_color="#000000", text_baseline="middle", text_align="center")
        glyph_sum_box = Ellipse(x="x", y="y", width="diameter", height="diameter", fill_color="#cab2d6", fill_alpha=0.4)
        glyph_commons = Ellipse(x="x", y="y", width="width", height="width", fill_color="#000000")

        # Create plot
        self.plot = Plot(
            min_height=500,
            width_policy='max',
            height_policy='max', 
            match_aspect=True)
        
        # Add block glyphs to plot
        glyph_system_box_renderer = self.plot.add_glyph(source_systems, glyph_system_box)
        glyph_sum_box_renderer = self.plot.add_glyph(source_sum, glyph_sum_box)
        glyph_system_text_renderer = self.plot.add_glyph(source_systems, glyph_system_text)
        glyph_sum_text_renderer = self.plot.add_glyph(source_sum, glyph_sum_text)
        glyph_commons_renderer = self.plot.add_glyph(source_commons, glyph_commons)

        # Add arrows
        for x_poly, y_poly in zip(x_polynomials, y_polynomials):
            glyph_arrow = Arrow(end=NormalHead(size=15, fill_color="#000000"),
                                x_start=x_poly[-2],
                                y_start=y_poly[-2],
                                x_end=x_poly[-1],
                                y_end=y_poly[-1],
                                line_width=0,
                                line_color = "#000000")
            self.plot.add_layout(glyph_arrow)

        # Add lines
        source_lines = ColumnDataSource(dict(xs=x_polynomials, ys=y_polynomials, output=output_lines))
        glyph_lines = MultiLine(xs="xs", ys="ys", line_color="#000000", line_width=3)
        glyph_lines_renderer = self.plot.add_glyph(source_lines, glyph_lines)

        # Tools
        node_hover_tool = HoverTool(
            line_policy="nearest", 
            tooltips="""
            <div>
                <div>
                    <b>Class Name:</b>
                    <i>@classname </i>
                </div>
                <div>
                    <b>States:</b>
                    <i>@states</i>
                </div>
                <div>
                    <b>Outputs:</b>
                    <i>@output</i>
                </div>
            </div>
                """,
            renderers=[glyph_system_box_renderer])

        # Should work from Bokeh 2.3.x
        sum_hover_tool = HoverTool(
            line_policy="nearest", 
            tooltips="""
                <div>
                    <div>
                        <b>Output:</b>
                        <i>@output </i>
                    </div>
                </div>
                    """,
            renderers=[glyph_sum_box_renderer, glyph_lines_renderer]
        )
        self.plot.add_tools(PanTool(), node_hover_tool, sum_hover_tool)

        if open:
            html = file_html(self.plot, CDN, "my plot")
            __write_to_browser__(html)
        

        


