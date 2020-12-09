"""
VISUALISATION LIBRARY (:mod: `nlcontrol.visualisation')
=======================================================

.. currentmodule:: nlcontrol.visualisation

Classes:
    * base:
        RendererBase : base class for visuals renderer.
        SystemRenderer : visual renderer for System blocks
        ParallelRenderer : visual renderer for parallel block schemes
        SeriesRenderer : visual renderer for Series block schemes

Functions:
    * file_management : 
        __clean_temp_folder__ : protected function to clean the nlcontrol files from the temporary folder.
        __write_to_browser__ : write html to browser window.
    * drawing_tools :
        draw_line : determine points to draw a line from one coordinate to another.
        generate_renderers_sources : from a renderer dict the bokeh sources are generated.
    * utils :
        pretty_print_dict : transforms a dict to a pretty formatted print version.



"""

from .base import RendererBase, SystemRenderer, ParallelRenderer, SeriesRenderer
from .file_management import __clean_temp_folder__, __write_to_browser__
from .drawing_tools import draw_line, generate_renderer_sources
from .utils import pretty_print_dict