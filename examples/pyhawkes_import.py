import sys
import os

# Appending directory above the current one to the sys.path
cur_dir = os.path.dirname(os.path.realpath(__file__))
split_dir = cur_dir.split('\\')
above_dir = '\\'.join(split_dir[:-1])
sys.path.append(above_dir)
import pyhawkes
