"""
VISUALISATION LIBRARY (:mod: `nlcontrol.visualisation')
=======================================================

.. currentmodule:: nlcontrol.visualisation

Classes:
    * base:
        RendererBase : base class for visuals renderer

Functions:
    * file_management : 
        __clean_temp_folder__ : protected function to clean the nlcontrol files from the temporary folder
    * drawing_tools :
        draw_line : determine points to draw a line from one coordinate to another


"""

from .base import RendererBase
from .file_management import __clean_temp_folder__