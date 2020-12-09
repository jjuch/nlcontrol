import json
import copy

def pretty_print_dict(dict_data: dict):
    json_data = json.dumps(dict_data, default=lambda o: '<not serializable>', indent=4)
    print(json_data)
    return json_data

def flatten_nodes(dict_data: dict, recursion_depth=0):
    flat_dict = dict()
    dict_data_copy = copy.deepcopy(dict_data)
    for node in dict_data_copy:
        cs = dict_data_copy[node]
        if 'nodes' in cs:
            cs_nodes = copy.deepcopy(cs['nodes'])
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
        

