"""
Seatbelt Detection - YOLOv11 All-in-One UI Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .train_utils import SeatbeltTrainer, download_roboflow_dataset
from .test_utils import SeatbeltTester
from .visual_utils import SeatbeltVisualizer
from .label_tool import SeatbeltLabeler

__all__ = [
    "SeatbeltTrainer",
    "SeatbeltTester", 
    "SeatbeltVisualizer",
    "SeatbeltLabeler",
    "download_roboflow_dataset"
]
