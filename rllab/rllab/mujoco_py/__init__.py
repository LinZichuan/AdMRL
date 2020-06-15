from .mjviewer import MjViewer
from .mjcore import MjModel
from .mjcore import register_license
import os
from .mjconstants import *

#register_license(os.environ['NICK_MUJOCO_KEY'])
register_license(os.environ['MUJOCO_LICENSE_PATH'])
