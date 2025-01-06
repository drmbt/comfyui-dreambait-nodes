print('''
██████╗ ██████╗ ███████╗ █████╗ ███╗   ███╗██████╗  █████╗ ██╗████████╗
██╔══██╗██╔══██╗██╔════╝██╔══██╗████╗ ████║██╔══██╗██╔══██╗██║╚══██╔══╝
██║  ██║██████╔╝█████╗  ███████║██╔████╔██║██████╔╝███████║██║   ██║   
██║  ██║██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║██╔══██╗██╔══██║██║   ██║   
██████╔╝██║  ██║███████╗██║  ██║██║ ╚═╝ ██║██████╔╝██║  ██║██║   ██║   
╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝   ╚═╝         
''')

import os

# Get the directory containing this file
DREAMBAIT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Point to the js directory at the root level
WEB_DIRECTORY = os.path.join(DREAMBAIT_ROOT, "js")

from .src.nodes_mappings import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]    