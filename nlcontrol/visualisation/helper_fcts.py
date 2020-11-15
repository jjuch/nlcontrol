def show():
    pass

class SystemBlock():
    def __init__(self, bbox, padx, pady, orientation='right'):
        self.bbox = bbox
        if orientation in ('right', 'left'):
            self.orientation=orientation
        else:
            error_text = "[SystemBlock]: The orientation is a string with the value 'left' or 'right'"
            raise ValueError(error_text)
        self.x0, self.y0, self.x1, self.y1 = self.get_text_corners()
        print('SystemBlock: ', self.get_port_coords(padx))
        self.in_coord, self.ex_coord = self.get_port_coords(padx)

    def get_text_corners(self):
        return self.bbox.x0, self.bbox.y0, self.bbox.x1, self.bbox.y1

    def get_port_coords(self, padx):
        left_coord = self.x0 - padx, (self.y1 - self.y0) / 2 + self.y0
        right_coord = self.x1 + padx, (self.y1 - self.y0) / 2 + self.y0
        if self.orientation == 'right':
            in_coord = left_coord
            out_coord = right_coord
        else:
            in_coord = right_coord
            out_coord = left_coord
        return left_coord, right_coord