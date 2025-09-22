import platform
import socket
import sys
from pathlib import Path
from subprocess import PIPE, Popen

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from MARS.utils.check_environments import check_environment_location, write_temp_bat_windows

#Check if pytorch environment is available, and if so get it's location.
pytorch_env = check_environment_location(env_name="pytorch_seg")

#This should work after packaged
#gui_path = Path(os.path.abspath(os.path.dirname(sys.argv[0]))).parent.parent.joinpath("morphology").joinpath("segmentation").joinpath("pytorch_segmentation")


if isinstance(pytorch_env, bool):
    print("Pytorch environment not found!")
else:
    pytorch_env = pytorch_env["Location"][0]
    pytorch_python = Path(pytorch_env).joinpath("python")
    three_class_script = Path(r"Z:\RyanLab\Projects\NStephens\git_repo\MARS\morphology\segmentation\pytorch_segmentation\3_class_gui.py")
    temp_batch = write_temp_bat_windows(batch_name="3_class", python_location=pytorch_python, script_location=three_class_script)
    process = Popen(f"{temp_batch}", shell=True, stdin=PIPE, stdout=PIPE)

