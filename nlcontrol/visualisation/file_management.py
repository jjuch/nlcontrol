import os
import glob
import tempfile

def __clean_temp_folder__():
    temp_folder = tempfile.gettempdir()
    for f in glob.glob(temp_folder + "\*.html"):
        if "nlcontrol" in f:
            os.remove(f)