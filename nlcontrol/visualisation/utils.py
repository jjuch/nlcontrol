import json
import copy

__all__ = ["pretty_print_dict", "flatten_nodes", "create_direction_vector", "flip_directions", "flip_block"]

def pretty_print_dict(dict_data: dict):
    json_data = json.dumps(dict_data, default=lambda o: '<not serializable>', indent=4)
    print(json_data)
    return json_data

def flatten_nodes(dict_data: dict, recursion_depth: int=0) -> dict:
    flat_dict = dict()
    dict_data_copy = copy.copy(dict_data)
    for node in dict_data_copy:
        cs = copy.copy(dict_data_copy[node])
        cs['depth'] = recursion_depth
        if 'nodes' in cs:
            cs_nodes = copy.copy(cs['nodes'])
            cs['nodes'] = dict()
            flat_nodes = flatten_nodes(cs_nodes, recursion_depth = recursion_depth + 1)
            flat_dict.update(flat_nodes)
        flat_dict.update({node: cs})
    return flat_dict

def create_direction_vector(direction: str) -> tuple:
    result = [0, 0]
    if direction == 'right':
        result[0] = 1
    elif direction == 'left':
        result[0] = -1
    elif direction == 'up':
        result[1] = 1
    elif direction == 'down':
        result[1] = -1
    else:
        error_text = "[Visualisation] A direction should be defined as 'left', 'right', 'up', 'down'. No other strings are allowed."
        raise ValueError(error_text)
    return tuple(result)

def flip_directions(old_directions: list or str):
    if type(old_directions) == list:
        new_directions = []
        for direc in old_directions:
            new_directions.append(flip_directions(direc))
    else:
        if old_directions == 'left':
            new_directions = 'right'
        elif old_directions == 'right':
            new_directions = 'left'
        else:
            new_directions = old_directions
    return new_directions


def flip_block(dict_data: dict, recursion_depth: int=0) -> dict:
    flipped_dict = dict()
    dict_data_copy = copy.copy(dict_data)
    for node in dict_data_copy:
        cs = copy.copy(dict_data_copy[node])
        cs['in_direction'] = flip_directions(cs['in_direction'])
        cs['out_direction'] = flip_directions(cs['out_direction'])
        if 'nodes' in cs:
            cs_nodes = copy.copy(cs['nodes'])
            cs['nodes'] = flip_block(cs_nodes, recursion_depth = recursion_depth + 1)
        flipped_dict.update({node: cs})
    return flipped_dict