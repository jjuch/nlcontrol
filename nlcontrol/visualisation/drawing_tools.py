import numpy as np

UNIT_LENGTH = 0.3

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
                    coord = (dx + dir1[0] * UNIT_LENGTH) * abs(dir1[0]) + coord1[0], (dy + dir1[1] * UNIT_LENGTH) * abs(dir1[1]) + coord1[1]
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
            coord = dir1[0] * UNIT_LENGTH + coord1[0], dir1[1] * UNIT_LENGTH + coord1[1]
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