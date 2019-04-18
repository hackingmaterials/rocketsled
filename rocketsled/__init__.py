"""
Rocketsled is an optimization suite "on rails" based on Scikit-learn and
FireWorks workflows.
"""

from rocketsled.control import MissionControl
from rocketsled.task import OptTask

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "../VERSION"), "r") as f:
    version = f.read()
__version__ = version

__author__ = "Alexander Dunn"
__email__ = "ardunn@lbl.gov"

