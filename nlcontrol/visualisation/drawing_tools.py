from nlcontrol.visualisation.utils import pretty_print_dict, flatten_nodes, create_direction_vector

import numpy as np
import uuid

from bokeh.models import ColumnDataSource

UNIT_WIRE_LENGTH = 0.3
UNIT_CHARACTER_LENGTH = 0.03
UNIT_CHARACTER_HEIGHT = 0.1


def draw_line(coord1, coord2, forbidden_direction=None, recursion_depth=0):
    # Initialize
    coordinate_list = []
    if recursion_depth == 0:
        coordinate_list.append((coord1[0], coord1[1]))
    dx = coord2[0] - coord1[0]
    dy = coord2[1] - coord1[1]

    # Helper function
    
    def get_direction(direction):
        if direction is not None:
            vertical_plane = 1 if direction in ('up', 'down') else 0
            if vertical_plane:
                new_direction = 1 if direction == 'up' else -1
                return (0, new_direction, vertical_plane)
            else:
                new_direction = 1 if direction == 'right' else -1
                return (new_direction, 0, vertical_plane)
        elif type(direction) == tuple:
            return direction
        else:
            return None

    def equal_directions(direction1, direction2):
        return (direction1[0] == direction2[0]) and (direction1[1] == direction2[1])

    def into_right_direction(direction1, direction2):
        return (direction1[0] == direction2[0]) or (direction1[1] == direction2[1])
    
    def invalid_direction(direction):
        return direction[0] == 0 and direction[1] == 0

    def same_plane(direction1, direction2):
        return direction1[2] == direction2[2]

    def next_coord_logic(coord1, coord2, dir1, dir2, right_direction):
        forbidden_direction = dir1
        if into_right_direction(dir1, right_direction):
            if same_plane(dir1, dir2):
                if dir1 != dir2:
                    # Pass the point by one unit length
                    coord = (dx + dir1[0] * UNIT_WIRE_LENGTH) * abs(dir1[0]) + coord1[0], (dy + dir1[1] * UNIT_WIRE_LENGTH) * abs(dir1[1]) + coord1[1]
                elif coord1[0] == coord2[0] or coord1[1] == coord2[1]:
                    # Go to the end node
                    coord = dx * abs(dir1[0]) + coord1[0], dy * abs(dir1[1]) + coord1[1]
                else:
                    # Get halfway the node
                    coord = dx / 2 * abs(dir1[0]) + coord1[0], dy / 2 * abs(dir1[1]) + coord1[1]
            elif into_right_direction(dir2, right_direction):
                # Go to the end node, but no connection yet
                coord = dx * abs(dir1[0]) + coord1[0], dy * abs(dir1[1]) + coord1[1]
            else:
                # Go halfway the node
                coord = dx / 2 * abs(dir1[0]) + coord1[0], dy / 2 * abs(dir1[1]) + coord1[1]
        else:
            # Move one unit length in the correct direction
            coord = dir1[0] * UNIT_WIRE_LENGTH + coord1[0], dir1[1] * UNIT_WIRE_LENGTH + coord1[1]
            forbidden_direction = (-1 * dir1[0], -1 * dir1[1], abs(dir1[1]))

        return coord, forbidden_direction

    # Main code
    coord1_dir = get_direction(coord1[2])
    coord2_dir = get_direction(coord2[2])
    right_direction = (int(np.sign(dx)), int(np.sign(dy)))
    

    if coord1_dir is not None:
        new_coord, forbidden_dir = next_coord_logic(coord1, coord2, coord1_dir, coord2_dir, right_direction)
    elif forbidden_direction is not None:
        new_direction1 = (right_direction[0], 0, 0)
        if same_plane(new_direction1, forbidden_direction) or (invalid_direction(new_direction1)):
            new_direction1 = (0, right_direction[1], 1)
        new_coord, forbidden_dir = next_coord_logic(coord1, coord2, new_direction1, coord2_dir, right_direction)
    else:
        if right_direction[1] != 0:
            new_direction1 = (0, right_direction[1], 1)
        else:
            new_direction1 = (right_direction[0], 0, 0)
        new_coord, forbidden_dir = next_coord_logic(coord1, coord2, new_direction1, coord2_dir, right_direction)

    
    if new_coord[0] != coord2[0] or new_coord[1] != coord2[1]:
        # Next recursion
        start_coord = (*new_coord, None)
        if recursion_depth <= 4:
            prev_coordinate_list = draw_line(start_coord, coord2, forbidden_dir, recursion_depth=recursion_depth + 1)
            # Process underlying recursion data
            prev_coordinate_list.append(new_coord)
            coordinate_list = prev_coordinate_list + coordinate_list
            if recursion_depth == 0:
                # Finale step of recursion
                coordinate_list.reverse()
                return coordinate_list
            else:
                return coordinate_list
        else:
            # Protection for unlimited recursion
            error_text = '[Visualisation.draw_line] The maximal recursion depth has been reached.'
            raise RecursionError(error_text)
    else:
        # maximal depth step in recursion
        coordinate_list.append(new_coord)
        return coordinate_list

    
def generate_system_renderer_info(self_obj, position=None, connect_from=[], connect_to=[]):
    if position is None:
        position = lambda x_off, y_off: (x_off, y_off)
    separator = ", "
    states_str = separator.join(\
        [str(state) for state in self_obj.states])
    info = {
        'type': 'system',
        'label': self_obj.block_name,
        'rel_position': position,
        'x_offset': 0,
        'y_offset': 0,
        'in_direction': 'right',
        'out_direction': 'right',
        'connect_to': connect_to,
        'connect_from': connect_from,
        'class_name': self_obj.__class__.__name__,
        'states': states_str,
        'output': ''
    }
    return info

def generate_summation_renderer_info(label='+', position=None, in_direction=['down', 'up'], out_direction='right', connect_to=[], connect_from=[], output=''):
    if position is None:
        position = lambda x_off, y_off: (x_off, y_off)
    info = {
        'type': 'summation',
        'label': label,
        'rel_position': position,
        'x_offset': 0,
        'y_offset': 0,
        'in_direction': in_direction,
        'out_direction': out_direction,
        'connect_to': connect_to,
        'connect_from': connect_from,
        'output': output
    }
    return info

def generate_common_node_renderer_info(position=None, in_direction='right', out_direction=['up', 'down'], connect_to=[], connect_from=[], output=''):
    if position is None:
        position = lambda x_off, y_off: (x_off, y_off)
    info = {
        'type': 'common',
        'rel_position': position,
        'x_offset': 0,
        'y_offset': 0,
        'connect_to': connect_to,
        'connect_from': connect_from,
        'output': output,
        'in_direction': in_direction,
        'out_direction': out_direction
    }
    return info


def generate_parallel_renderer_info(self_obj, systems, output=''):
    number_of_blocks = 4
    id_list = [uuid.uuid4().hex for _ in range(number_of_blocks)]
    
    position = lambda x_off, y_off: (x_off, y_off)
    info = {
        'type': 'parallel',
        'label': self_obj.block_name,
        'rel_position': position,
        'x_offset': 0,
        'y_offset': 0,
        'in_direction': 'right', 
        'out_direction': 'right',
        'connect_to': [],
        'connect_from': [],
        'nodes': dict(), 
        'output': output
    }
    nodes_dict = info['nodes']

    # Add system nodes
    for i, system in enumerate(systems):
        # i has no pointer, therefore declared as a default parameter
        position = lambda x_off, y_off, i=i: (0.5 + x_off, 0.5 * i + y_off)
        system_dict = generate_system_renderer_info(
            system, 
            position=position,
            connect_to=[id_list[2]], 
            connect_from=[id_list[3]])
        new_dict = {id_list[i]: system_dict}
        nodes_dict.update(new_dict)
    
    # Add summation node
    position = lambda x_off, y_off: (1 + x_off, 0.25 + y_off)
    summation_dict = generate_summation_renderer_info(
        position=position, 
        connect_from=[id_list[0], id_list[1]])
    new_dict = {id_list[2]: summation_dict}
    nodes_dict.update(new_dict)

    # Add input_node
    position = lambda x_off, y_off: (x_off, 0.25 + y_off)
    input_node_dict = generate_common_node_renderer_info(
        position=position, 
        connect_to=[id_list[0], id_list[1]])
    new_dict = {id_list[3]: input_node_dict}
    nodes_dict.update(new_dict)

    return info


def generate_renderer_sources(renderer_info, recursion_depth=0):
    x_sys = []
    y_sys = []
    x_sum = []
    y_sum = []
    x_comm = []
    y_comm = []
    text_sys = []
    text_sum = []
    width_sys = []
    height_sys = []
    diameter = []
    radius = []
    width_comm = []
    classname = []
    states = []
    output_sys = []
    output_sum = []
    

    for node in renderer_info:
        cs = renderer_info[node]
        if cs['type'] == 'system':
            # Set coordinate of center of block
            x_sys.append(cs['position'][0])
            y_sys.append(cs['position'][1])
            # Set dimensions
            text_sys.append(cs['label'])
            label_length = len(cs['label'])
            width_sys.append(UNIT_CHARACTER_LENGTH * (label_length + 2))
            height_sys.append(UNIT_CHARACTER_HEIGHT)
            left_coord = (x_sys[-1] - width_sys[-1] / 2, y_sys[-1])
            right_coord = (x_sys[-1] + width_sys[-1] / 2, y_sys[-1])
            # Get input and output coordinates
            if cs['in_direction'] == 'right':
                cs['in_pos'] = left_coord
                cs['out_pos'] = right_coord
            else:
                cs['in_pos'] = right_coord
                cs['out_pos'] = left_coord
            # Set information
            classname.append(cs['class_name'])
            states.append(cs['states'])
            output_sys.append(cs['output'])
        elif cs['type'] == 'summation':
            # Set coordinate of center of block
            x_sum.append(cs['position'][0])
            y_sum.append(cs['position'][1])
            # Set dimensions
            output_sum.append(cs['output'])
            text_sum.append(cs['label'])
            label_length = len(cs['label'])
            diameter.append(UNIT_CHARACTER_LENGTH * (label_length + 2))
            cs['diameter'] = diameter[-1]
            radius.append(diameter[-1] / 2)
            # Get input coordinate
            in_coord = []
            for direc in cs['in_direction']:
                if direc == 'up':
                    in_coord_temp = (x_sum[-1], y_sum[-1] - diameter[-1] / 2)
                elif direc == 'down':
                    in_coord_temp = (x_sum[-1], y_sum[-1] + diameter[-1] / 2)
                elif direc == 'left':
                    in_coord_temp = (x_sum[-1] - diameter[-1] / 2, y_sum[-1])
                elif direc == 'right':
                    in_coord_temp = (x_sum[-1] + diameter[-1] / 2, y_sum[-1])
                else:
                    error_text = "[Visualisation.drawing_tools] The directions can only be defined by the strings 'up', 'down', 'left', and 'right'."
                    raise ValueError(error_text)
                in_coord.append(in_coord_temp)
            cs['in_pos'] = in_coord
            # Get output coordinate
            if cs['out_direction'] == 'up':
                cs['out_pos'] = (x_sum[-1], y_sum[-1] - diameter[-1] / 2)
            elif cs['out_direction'] == 'down':
                cs['out_pos'] = (x_sum[-1], y_sum[-1] + diameter[-1] / 2)
            elif cs['out_direction'] == 'left':
                cs['out_pos'] = (x_sum[-1] - diameter[-1] / 2, y_sum[-1])
            elif cs['out_direction'] == 'right':
                cs['out_pos'] = (x_sum[-1] + diameter[-1] / 2, y_sum[-1])
            else:
                error_text = "[Visualisation.drawing_tools] The directions can only be defined by the strings 'up', 'down', 'left', and 'right'."
                raise ValueError(error_text)
        elif cs['type'] == 'common':
            x_comm.append(cs['position'][0])
            y_comm.append(cs['position'][1])
            width_comm.append(UNIT_CHARACTER_LENGTH)
            cs['diameter'] = width_comm[-1]
            # Set input and output coordinates
            if cs['in_direction'] == 'up':
                in_coord = (x_comm[-1], y_comm[-1] + width_comm[-1] / 2)
            elif cs['in_direction'] == 'down':
                in_coord = (x_comm[-1], y_comm[-1] - width_comm[-1] / 2)
            elif cs['in_direction'] == 'left':
                in_coord = (x_comm[-1] + width_comm[-1] / 2, y_comm[-1])
            elif cs['in_direction'] == 'right':
                in_coord = (x_comm[-1] - width_comm[-1] / 2, y_comm[-1])
            else:
                error_text = "[Visualisation.drawing_tools] The directions can only be defined by the strings 'up', 'down', 'left', and 'right'."
                raise ValueError(error_text)
            cs['in_pos'] = in_coord
            cs['out_pos'] = cs['position']
        elif 'nodes' in cs:
            cs_nodes = cs['nodes']
            # Do first the recursion to add 'in_pos' and 'out_pos' to the renderer info of the children before it can be used in parent nodes
            sys_dict, sum_dict, comm_dict = generate_renderer_sources(cs_nodes, recursion_depth=recursion_depth + 1)
            # print(sys_dict)
            # append recursive system data
            x_sys.extend(sys_dict['x'])
            y_sys.extend(sys_dict['y'])
            text_sys.extend(sys_dict['text'])
            width_sys.extend(sys_dict['width'])
            height_sys.extend(sys_dict['height'])
            classname.extend(sys_dict['classname'])
            states.extend(sys_dict['states'])
            output_sys.extend(sys_dict['output'])
            # extend recursive sum data
            x_sum.extend(sum_dict['x'])
            y_sum.extend(sum_dict['y'])
            text_sum.extend(sum_dict['text'])
            diameter.extend(sum_dict['diameter'])
            radius.extend(sum_dict['radius'])
            output_sum.extend(sum_dict['output'])
            # extend recursive common node data
            x_comm.extend(comm_dict['x'])
            y_comm.extend(comm_dict['y'])
            width_comm.extend(comm_dict['width'])

            #TODO: Should move to generate_parallel_renderer_info
            if cs['type'] == 'parallel':
                for parallel_node in cs_nodes:
                    if cs_nodes[parallel_node]['type'] == 'common':
                        cs['in_pos'] = cs_nodes[parallel_node]['position']
                    elif cs_nodes[parallel_node]['type'] == 'summation':
                        cs['out_pos'] = cs_nodes[parallel_node]['out_pos']
            

    if recursion_depth == 0:
        source_systems = ColumnDataSource(dict(x=x_sys, y=y_sys, text=text_sys, width=width_sys, height=height_sys, classname=classname, states=states, output=output_sys))
        source_sum = ColumnDataSource(dict(x=x_sum, y=y_sum, text=text_sum, diameter=diameter, radius=radius, output=output_sum))
        source_commons = ColumnDataSource(dict(x=x_comm, y=y_comm, width=width_comm))
        
        return source_systems, source_sum, source_commons
    else:
        sys_dict = {'x': x_sys, 'y': y_sys, 'text': text_sys, 'width': width_sys, 'height': height_sys, 'output': output_sys, 'states': states, 'classname': classname}
        sum_dict = {'x': x_sum, 'y': y_sum, 'text': text_sum, 'diameter': diameter, 'radius': radius, 'output': output_sum}
        comm_dict = {'x': x_comm, 'y': y_comm, 'width': width_comm}
        return sys_dict, sum_dict, comm_dict


def generate_connection_coordinates(renderer_info):
    x_polynomials = []
    y_polynomials = []
    output = []

    new_renderer_info = flatten_nodes(renderer_info)
    pretty_print_dict(new_renderer_info)
    
    for node in new_renderer_info:
        cs = new_renderer_info[node]
        polyn_x_coords = []
        polyn_y_coords = []
        print("================Test:")
        pretty_print_dict(cs)
        if 'diameter' in cs:
            width = cs['diameter']
        else:
            width = abs(cs['out_pos'][0] - cs['in_pos'][0])
        # Add an arrow to an empty input node without children
        if (len(cs['connect_from']) == 0) and ('nodes' not in cs):
            print('Yes: ', width)
            # End coordinate
            polyn_x_coords.append(cs['in_pos'][0])
            polyn_y_coords.append(cs['in_pos'][1])
            # Begin coordinate
            sign = create_direction_vector(cs['in_direction'])
            polyn_x_coords.append(cs['in_pos'][0] - sign[0] * width)
            polyn_y_coords.append(cs['in_pos'][1] - sign[1] * width)

            # Add flipped coordinates
            polyn_x_coords.reverse()
            polyn_y_coords.reverse()
            x_polynomials.append(polyn_x_coords)
            y_polynomials.append(polyn_y_coords)
            # For index consistency
            output.append(None)

            polyn_x_coords = []
            polyn_y_coords = []

        # Add an arrow to an empty output node without children
        if (len(cs['connect_to']) == 0) and ('nodes' not in cs):
            # Begin coordinate
            polyn_x_coords.append(cs['out_pos'][0])
            polyn_y_coords.append(cs['out_pos'][1])
            # End coordinate
            sign = create_direction_vector(cs['out_direction'])
            polyn_x_coords.append(cs['out_pos'][0] + sign[0] * width)
            polyn_y_coords.append(cs['out_pos'][1] + sign[1] * width)

            x_polynomials.append(polyn_x_coords)
            y_polynomials.append(polyn_y_coords)
            output.append(cs['output'])
        else:
            for connection in cs['connect_to']:
                end_block = new_renderer_info[connection]
                # Set start coordinate
                start_coord = []
                start_coord.extend(cs['out_pos'])
                if type(cs['out_direction']) == list:
                    #TODO: FIX: should depend on direction of arrows
                    if cs['position'][1] < end_block['position'][1]:
                        start_coord.append(cs['out_direction'][0])
                    else:
                        start_coord.append(cs['out_direction'][1])
                else:
                    start_coord.append(cs['out_direction'])
                # Get end coordinate - initialize
                stop_coord = []
                # If the end block has multiple inputs, select the correct one.
                if type(end_block['in_direction']) == list:
                    #TODO: FIX: should depend on direction of arrows
                    if cs['position'][1] > end_block['position'][1]:
                        stop_coord.extend(end_block['in_pos'][0])
                        stop_coord.append(end_block['in_direction'][0])
                    else:
                        stop_coord.extend(end_block['in_pos'][1])
                        stop_coord.append(end_block['in_direction'][1])
                else:
                    stop_coord.extend(end_block['in_pos'])
                    stop_coord.append(end_block['in_direction'])

                # Draw line between start en stop coordinate
                print(start_coord, " - ", stop_coord)
                line_coordinates = draw_line(start_coord, stop_coord)
                # Format coordinate lists
                for line_coord in line_coordinates:
                    polyn_x_coords.append(line_coord[0])
                    polyn_y_coords.append(line_coord[1])
                x_polynomials.append(polyn_x_coords)
                y_polynomials.append(polyn_y_coords)
                output.append(cs['output'])
                polyn_x_coords = []
                polyn_y_coords = []

    return x_polynomials, y_polynomials, output

                

        



            



if __name__ == "__main__":
    #####################################
    # Tester draw_lines                 #
    #####################################
    # Case 1
    print("Case 1:")
    coord1 = (1, 7, 'down')
    coord2 = (5, 1, 'down')
    res1 = draw_line(coord1, coord2)
    print(res1)

    # Case 2
    print("\n\nCase 2:")
    coord1 = (1, 7, 'down')
    coord2 = (5, 1, 'up')
    res2 = draw_line(coord1, coord2)
    print(res2)

    # Case 3
    print("\n\nCase 3:")
    coord1 = (1, 2, 'up')
    coord2 = (5, 6, 'right')
    res3 = draw_line(coord1, coord2)
    print(res3)

    # Case 4
    print("\n\nCase 4:")
    coord1 = (1, 2, 'up')
    coord2 = (5, 6, 'left')
    res4 = draw_line(coord1, coord2)
    print(res4)
    
    # Case 5
    print("\n\nCase 5:")
    coord1 = (1, 2, 'down')
    coord2 = (5, 6, 'up')
    res5 = draw_line(coord1, coord2)
    print(res5)

    # Case 6
    print("\n\nCase 6:")
    coord1 = (1, 2, 'down')
    coord2 = (5, 6, 'right')
    res6 = draw_line(coord1, coord2)
    print(res6)

    # Case 7
    print("\n\nCase 7:")
    coord1 = (1, 2, 'right')
    coord2 = (5, 2, 'right')
    res7 = draw_line(coord1, coord2)
    print(res7)