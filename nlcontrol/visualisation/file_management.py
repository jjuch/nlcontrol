import os
import glob
import tempfile, webbrowser
import json

def __clean_temp_folder__():
    temp_folder = tempfile.gettempdir()
    for f in glob.glob(temp_folder + "\*.html"):
        if "nlcontrol" in f:
            os.remove(f)

def __write_to_browser__(html, clean_temp=True):
    if clean_temp:
        __clean_temp_folder__()
    with tempfile.TemporaryFile(mode='w+t', prefix="nlcontrol_", suffix=".html", delete=False) as f:
        f.write(html)
        f.flush()
        browser = webbrowser.get()
        browser.open_new(f.name)