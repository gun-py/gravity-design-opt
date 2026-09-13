# alias for the shipped l2d/name_gen.py so tools can import without touching sys.path much
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "l2d"))
from name_gen import *  # noqa
