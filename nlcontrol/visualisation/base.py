import ctypes
import sys
import subprocess

class RendererBase():
    def __init__(self):
        self.screen_info = self.__get_screen_info__()
        self.plot = None
        self.plot_dict = dict()
        self.renderer_info = self.__init_renderer_info__()

    def __get_screen_info__(self) -> dict:
        os_type = sys.platform
        print(os_type)
        try:
            if os_type == 'win32':
                user32 = ctypes.windll.user32
                screen_size = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
            elif os_type == 'linux':
                output = subprocess.Popen('xrandr | grep "\*" | cut -d" " -f4',shell=True, stdout=subprocess.PIPE).communicate()[0]
                screen_size = tuple(output.split()[0].split(b'x'))
            else:
                screen_size = (800, 700)
        except Exception as e:
            print(e)
            
        screen_ratio = screen_size[1] / screen_size[0]
        return {'size': screen_size, 'ratio': screen_ratio}

    
    def __init_renderer_info__(self):
        info = dict()
        return info
        

    def show(self):
        print("Showing the system {} with name '{}'".format(type(self), self.name))
        print('Screen info: ', self.screen_info)
