def draw_line(coord1, coord2):
    dx = coord1[0] - coord2[0]
    dy = coord1[1] - coord2[1]
    
    def get_direction(dir_str):
        vertical_plane = 1 if dir_str in ('up', 'down') else 0
        if vertical_plane:
            direction = 1 if dir_str == 'up' else -1
            return (0, direction, vertical_plane)
        else:
            direction = 1 if dir_str == 'right' else -1
            return (direction, 0, vertical_plane)
    coord1_dir = get_direction(coord1[2])
    coord2_dir = get_direction(coord2[2])
    
    
    # Number of points needed in between
    steps = 2 if coord1_dir == coord2_dir else 1
    coords = []


    

def draw_nearest_in_direction(coord1, coord2, direction):
    if direction == 'up':
        pass