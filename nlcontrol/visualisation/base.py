from nlcontrol.visualisation.file_management import __write_to_browser__
from nlcontrol.visualisation.drawing_tools import draw_line, generate_relative_positions, generate_renderer_sources, generate_connection_coordinates, update_renderer_info, generate_summation_renderer_info, generate_common_node_renderer_info
from nlcontrol.visualisation.utils import pretty_print_dict, flip_block

from bokeh.io import show, output_file, curdoc
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.models import (Arrow, ColumnDataSource, Ellipse, HoverTool, MultiLine, NormalHead, PanTool, Plot, Range1d, Rect, Text)

import uuid

__all__ = ["RendererBase", "SystemRenderer", "ParallelRenderer", "SignalRenderer", "ClosedLoopRenderer"]

FONT_SIZE_IN_PIXELS = 15
x_offset, y_offset = 0, 0

class RendererBase(object):
    def __init__(self, system_obj, **kwargs):
        self.plot = None
        self.plot_dict = dict()
        self.system_obj = system_obj
        self.renderer_info = None

    def __init_renderer_info__(self, generate_renderer_info_function, block_type="system", **kwargs):
        unique_id = uuid.uuid4().hex
        info = {unique_id : dict()}
        info_id = info[unique_id]
        if 'block_type' in kwargs:
            block_type = kwargs['block_type']
        print(block_type)
        info_dict = generate_renderer_info_function(self.system_obj, **kwargs)
        info_id.update(info_dict)
        return info

    def get_dimensions(self, renderer_info=None):
        # Get width and heights from each subsystem in the parallel block scheme. The order in the vectors is top_system, bottom_system, summation, common node.
        if renderer_info is None:
            renderer_info = self.renderer_info
        try:
            # print(list(renderer_info.keys()))
            uuid_parallel = list(renderer_info.keys())[0]
            parent = renderer_info[uuid_parallel]
            # print("==== parent:")
            # pretty_print_dict(parent)
            children_nodes = parent['nodes']
        except:
            error_text = "[Visualisation.RendererBase] Supply the single parent as renderer_info. No other nodes should be included."
            raise ValueError(error_text)
        widths = []
        heights = []
        for child_id in children_nodes:
            child_node = children_nodes[child_id]
            # print("======= width?")
            # pretty_print_dict(child_node)
            if 'diameter' in child_node:
                widths.append(child_node['diameter'])
                heights.append(child_node['diameter'])
            else:
                widths.append(child_node['width'])
                heights.append(child_node['height'])
        return widths, heights


    def show(self, open_browser=True):
        print("Showing the system {} with name '{}'".format(type(self.system_obj), self.system_obj.name))
        generate_relative_positions(self.renderer_info)
        # print("===== renderer_info: ")
        # pretty_print_dict(self.renderer_info)
        
        source_systems, source_sum, source_commons = generate_renderer_sources(self.renderer_info)
        # print("===== renderer_info: ")
        # pretty_print_dict(self.renderer_info)

        x_polynomials, y_polynomials, output_lines = generate_connection_coordinates(self.renderer_info)

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

        if open_browser:
            html = file_html(self.plot, CDN, "my plot")
            __write_to_browser__(html)
        

class SystemRenderer(RendererBase):
    def __init__(self, system_obj, **kwargs):
        super().__init__(system_obj, **kwargs)
        self.renderer_info = self.__init_renderer_info__(**kwargs)

    def __init_renderer_info__(self, **kwargs):
        return super().__init_renderer_info__(self.generate_system_renderer_info, **kwargs)

    def generate_system_renderer_info(self, system_obj, position=None, connect_from=[], connect_to=[]):
        if position is None:
            position = lambda x_off, y_off, width: (x_off + width / 2, y_off)
        if system_obj.states is None:
            states_str = ''
        else:
            separator = ", "
            states_str = separator.join(\
                [str(state) for state in system_obj.states])
        info = {
            'type': 'system',
            'label': system_obj.block_name,
            'rel_position': position,
            'in_direction': 'right',
            'out_direction': 'right',
            'connect_to': connect_to,
            'connect_from': connect_from,
            'class_name': system_obj.__class__.__name__,
            'states': states_str,
            'output': ''
        }
        return info

    def get_dimensions(self, **kwargs):
        error_text = "[Visualisation.SystemRenderer] The SystemRenderer class has no useful module called `get_dimensions'."
        raise ModuleNotFoundError(error_text)


    def get_position_function_arguments(self, parent_offsets, renderer_info=None, unit_block_space=0.5):
        return parent_offsets[0], parent_offsets[1], renderer_info['width']

    
    def set_coordinates(self, current_element=None):
        if current_element is None:
            current_element = self.renderer_info
        for current_id in current_element.keys():
            current_data = current_element[current_id]
            current_data['position'] = current_data['rel_position'](current_data['x_offset'], current_data['y_offset'])
            if 'nodes' in current_data:
                self.set_coordinates(current_element=current_data['nodes'])
        # print(self.renderer_info)



class ParallelRenderer(RendererBase):
    def __init__(self, system_obj, **kwargs):
        super().__init__(system_obj)
        self.renderer_info = self.__init_renderer_info__(**kwargs)

    def __init_renderer_info__(self, block_type="parallel", **kwargs):
        if 'systems' not in kwargs:
            error_text = "[visualisation.RendererBase] In the case of a 'parallel' block_type a key 'systems' should be supplied."
            raise AttributeError(error_text)
        return super().__init_renderer_info__(self.generate_parallel_renderer_info, **kwargs)

    def generate_parallel_renderer_info(self, system_obj, systems, output=''):
        number_of_blocks = 4
        id_list = [uuid.uuid4().hex for _ in range(number_of_blocks)]
        
        position = lambda x_off, y_off: (x_off, y_off)
        
        info = {
            'type': 'parallel',
            'label': system_obj.block_name,
            'rel_position': position,
            'x_offset': 0,
            'y_offset': 0,
            'in_direction': 'right', 
            'out_direction': 'right',
            'connect_to': [],
            'connect_from': [],
            'nodes': dict(), 
            'output': output,
            'renderer': self
        }
        nodes_dict = info['nodes']

        # Add system nodes
        sign = [1, -1] # One system above and one below reference
        for i, system in enumerate(systems):
            sys_renderer_info = system.renderer.renderer_info
            system_id = list(sys_renderer_info.keys())[0]
            # i has no pointer, therefore declared as a default parameter
            if 'nodes' in system.renderer.renderer_info[system_id]:
                position = lambda x_off, y_off, widths, heights, unit_block_space=0.5, i=i: (
                    unit_block_space + x_off, 
                    sign[i] * (unit_block_space + heights[i])/2 + y_off
                )
            else:
                position = lambda x_off, y_off, widths, heights, unit_block_space=0.5, i=i: (
                    unit_block_space + widths[i] / 2 + x_off, 
                    sign[i] * (unit_block_space + heights[i])/2 + y_off
                )
            new_renderer_info = update_renderer_info(
                system.renderer.renderer_info,
                id_list[i],
                rel_position=position,
                connect_to=[id_list[2]], 
                connect_from=[id_list[3]],
                renderer=system.renderer)
            nodes_dict.update(new_renderer_info)
        
        # Add summation node
        position = lambda x_off, y_off, widths, heights, unit_block_space=0.5: (
            2 * unit_block_space + max(widths[0:2]) + x_off,
            y_off
        )
        summation_dict = generate_summation_renderer_info(
            position=position, 
            connect_from=[id_list[0], id_list[1]])
        new_dict = {id_list[2]: summation_dict}
        nodes_dict.update(new_dict)

        # Add input_node (is origin)
        position = lambda x_off, y_off, widths, heights, unit_block_space=0.5: (x_off, y_off)
        input_node_dict = generate_common_node_renderer_info(
            position=position, 
            connect_to=[id_list[0], id_list[1]])
        new_dict = {id_list[3]: input_node_dict}
        nodes_dict.update(new_dict)

        return info

    def get_position_function_arguments(self, parent_offsets, renderer_info=None, unit_block_space=0.5):
        if renderer_info is None:
            renderer_info = self.renderer_info
        widths, heights = self.get_dimensions(renderer_info=renderer_info)
        return parent_offsets[0], parent_offsets[1], widths, heights, unit_block_space


    def calculate_dimension(self, renderer_info=None, unit_block_space=0.5):
        if renderer_info is None:
            renderer_info = self.renderer_info
        # Get width and heights of children nodes
        widths, heights = self.get_dimensions(renderer_info=renderer_info)
        
        # Estimate width and height of parent node
        width = widths[3] + unit_block_space + max(widths[0], widths[1]) + unit_block_space + widths[2]
        height = heights[0] + unit_block_space + heights[1]
        return width, height


class SeriesRenderer(RendererBase):
    def __init__(self, system_obj, **kwargs):
        super().__init__(system_obj)
        self.renderer_info = self.__init_renderer_info__(**kwargs)

    def __init_renderer_info__(self, block_type="series", **kwargs):
        if 'systems' not in kwargs:
            error_text = "[RendererBase] In the case of a 'series' block_type a keyword argument 'systems' should be supplied."
            raise AttributeError(error_text)
        return super().__init_renderer_info__(self.generate_series_renderer_info, **kwargs)


    def generate_series_renderer_info(self, system_obj, systems, output=''):
        number_of_blocks = 2
        id_list = [uuid.uuid4().hex for _ in range(number_of_blocks)]

        position = lambda x_off, y_off: (x_off, y_off)

        info = {
            'type': 'series',
            'label': system_obj.block_name,
            'rel_position': position,
            'x_offset': 0,
            'y_offset': 0,
            'in_direction': 'right', 
            'out_direction': 'right',
            'connect_to': [],
            'connect_from': [],
            'nodes': dict(), 
            'output': output,
            'renderer': self
        }
        nodes_dict = info['nodes']

        # Add system nodes
        for i, system in enumerate(systems):
            sys_renderer_info = system.renderer.renderer_info
            system_id = list(sys_renderer_info.keys())[0]
            # i has no pointer, therefore declared as a default parameter
            if 'nodes' in system.renderer.renderer_info[system_id]:
                position = lambda x_off, y_off, widths, unit_block_space=0.5, i=i: (x_off + i * unit_block_space + sum(widths[:i]), y_off)
            else:
                position = lambda x_off, y_off, widths, unit_block_space=0.5, i=i: (x_off + widths[i] / 2 + i * unit_block_space + sum(widths[:i]), y_off)
            connect_to = []
            connect_from = []
            if i < len(systems) - 1:
                connect_to.append(id_list[i + 1])
            if i > 0:
                connect_from.append(id_list[i - 1])
                
            new_renderer_info = update_renderer_info(
                system.renderer.renderer_info,
                id_list[i],
                rel_position=position,
                connect_to=connect_to, 
                connect_from=connect_from)
            nodes_dict.update(new_renderer_info)

        return info

    def get_position_function_arguments(self, parent_offsets, renderer_info=None, unit_block_space=0.5):
        if renderer_info is None:
            renderer_info = self.renderer_info
        widths, heights = self.get_dimensions(renderer_info=renderer_info)
        return parent_offsets[0], parent_offsets[1], widths, unit_block_space

    def calculate_dimension(self, renderer_info=None, unit_block_space=0.5):
        # Get width and heights from each subsystem in the series block scheme. The order in the vectors is left_system, right_system.
        if renderer_info is None:
            renderer_info = self.renderer_info

        # Get all widths and heights of children nodes
        widths, heights = self.get_dimensions(renderer_info=renderer_info)

        # Estimate width and height of parent node
        width = widths[0] + unit_block_space + widths[1]
        height = max(heights)
        return width, height


class SignalRenderer(SystemRenderer):
    # def __init__(self, system_obj, **kwargs):
    #     super().__init__(system_obj, **kwargs)
    #     self.renderer_info = self.__init_renderer_info__(self, **kwargs)

    def generate_system_renderer_info(self, system_obj, position=None, connect_from=[None], connect_to=[]):
        return super().generate_system_renderer_info(
            system_obj, 
            position=position,
            connect_from=connect_from,
            connect_to=connect_to)
        
        
class ClosedLoopRenderer(RendererBase):
    def __init__(self, system_obj, **kwargs):
        super().__init__(system_obj)
        self.renderer_info = self.__init_renderer_info__(system_obj, **kwargs)

    def __init_renderer_info__(self, system_obj, block_type="closedloop", **kwargs):
        print("kwargs: ", kwargs)
        if 'forward_sys' not in kwargs:
            error_text = "[visualisation.RendererBase] In the case of a 'closedloop' block_type a key 'forward_sys' should be supplied." 
            raise AttributeError(error_text)
        if 'backward_sys' not in kwargs:
            error_text = "[visualisation.RendererBase] In the case of a 'closedloop' block_type a key 'backward_sys' should be supplied." 
            raise AttributeError(error_text)

        
        return super().__init_renderer_info__(self.generate_closed_loop_renderer_info, **kwargs)


    def generate_closed_loop_renderer_info(self, system_obj, forward_sys=None, backward_sys=None, output=''):
        number_of_blocks = 4
        id_list = [uuid.uuid4().hex for _ in range(number_of_blocks)]

        position = lambda x_off, y_off: (x_off, y_off)

        info = {
            'type': 'closedloop',
            'label': system_obj.block_name,
            'rel_position': position,
            'x_offset': 0,
            'y_offset': 0,
            'in_direction': 'right',
            'out_direction': 'right',
            'connect_to': [],
            'connect_from': [],
            'nodes': dict(),
            'output': output,
            'renderer': self
        }
        nodes_dict = info['nodes']

        # Add forward system node
        fwd_sys_renderer_info = forward_sys.renderer.renderer_info
        fwd_system_id = list(fwd_sys_renderer_info.keys())[0]

        if 'nodes' in forward_sys.renderer.renderer_info[fwd_system_id]:
            fwd_position = lambda x_off, y_off, widths, heights, unit_block_space=0.5: (
                widths[2] / 2 + unit_block_space + x_off,
                y_off
            )
        else:
            fwd_position = lambda x_off, y_off, widths, heights, unit_block_space=0.5: (
                widths[2] / 2 + unit_block_space + widths[0] / 2 + x_off,
                y_off
            )
        new_fwd_renderer_info = update_renderer_info(
            forward_sys.renderer.renderer_info,
            id_list[0],
            rel_position=fwd_position,
            connect_to=[id_list[3]],
            connect_from=[id_list[2]],
            renderer=forward_sys.renderer
        )
        nodes_dict.update(new_fwd_renderer_info)

        # Add backward system node
        bwd_position = lambda x_off, y_off, widths, heights, unit_block_space=0.5: (
            widths[2] / 2 + unit_block_space + widths[0] / 2 + x_off,
            -heights[0] / 2 - unit_block_space - heights[1] / 2 + y_off
        )
        connect_to = [id_list[2]]
        connect_from = [id_list[3]]
        if backward_sys is not None:
            bwd_sys_renderer_info = backward_sys.renderer.renderer_info
            bwd_system_id = list(bwd_sys_renderer_info.keys())[0]

            if 'nodes' in backward_sys.renderer.renderer_info[bwd_system_id]:
                bwd_position = lambda x_off, y_off, widths, heights, unit_block_space=0.5: (
                    widths[2] / 2 + unit_block_space + x_off,
                    -heights[0] / 2 - unit_block_space - heights[1] / 2 + y_off
                )
            
            new_bwd_renderer_info = update_renderer_info(
                flip_block(backward_sys.renderer.renderer_info),
                id_list[1],
                rel_position=bwd_position,
                connect_to=connect_to,
                connect_from=connect_from,
                renderer=backward_sys.renderer
            )
        else:
            new_bwd_renderer_info = generate_common_node_renderer_info(
                in_direction='left',
                out_direction='left',
                position=bwd_position,
                connect_to=connect_to,
                connect_from=connect_from
            )

        nodes_dict.update(new_bwd_renderer_info)

        # Add summation node (is origin)
        position = lambda x_off, y_off, widths, heights, unit_block_space=0.5: (x_off, y_off)
        connect_from = [id_list[3]] if backward_sys is None\
            else [id_list[1]]
            
        summation_dict = generate_summation_renderer_info(
            position=position,
            connect_to=[id_list[0]],
            connect_from=connect_from,
            in_direction=['down', 'right']
        )
        new_dict = {id_list[2]: summation_dict}
        nodes_dict.update(new_dict)

        # Add output node
        position = lambda x_off, y_off, widths, heights, unit_block_space=0.5: (
            widths[2] / 2 + 2 * unit_block_space + max(widths[0:2]) + x_off,
            y_off
        )
        connect_to = [id_list[2]] if backward_sys is None\
            else [id_list[1]]
        output_node_dict = generate_common_node_renderer_info(
            position=position,
            connect_to=connect_to,
            connect_from=[id_list[0]],
            out_direction=['down', 'right']
        )
        new_dict = {id_list[3]: output_node_dict}
        nodes_dict.update(new_dict)

        return info

    def get_position_function_arguments(self, parent_offsets, renderer_info=None, unit_block_space=0.5):
        if renderer_info is None:
            renderer_info = self.renderer_info
        widths, heights = self.get_dimensions(renderer_info=renderer_info)
        return parent_offsets[0], parent_offsets[1], widths, heights, unit_block_space

    def calculate_dimension(self, renderer_info=None, unit_block_space=0.5):
        if renderer_info is None:
            renderer_info = self.renderer_info
        # Get width and heights of children nodes
        widths, heights = self.get_dimensions(renderer_info=renderer_info)
        print(widths, " - ", heights)

        # Estimate width and heigh of parent node
        width = widths[2] + unit_block_space + max(widths[0:2]) + unit_block_space + widths[3]
        height = heights[0] + unit_block_space + heights[1]
        return width, height        