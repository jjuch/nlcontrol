import json

def pretty_print_dict(dict_data):
    json_data = json.dumps(dict_data, default=lambda o: '<not serializable>', indent=4)
    print(json_data)
    return json_data