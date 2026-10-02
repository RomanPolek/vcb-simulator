import sys
from pathlib import Path

VCB_SIMULATOR_ROOT = Path(__file__).resolve().parent

# these settings should never really change
LOGIC_LAYER_INDEX = 0
VCB_ROW_WIDTH = 2048


sys.path.insert(0, str(VCB_SIMULATOR_ROOT))
