import os
import sys

def add_source_to_sys_path():
    source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if source_path not in sys.path:
        sys.path.append(source_path)

