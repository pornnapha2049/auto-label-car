from .pipeline import CarSegmenter, DEFAULT_CATEGORIES, CLASS_COLORS, DEFAULT_CATEGORIES as DEFAULT_VEHICLE_CLASSES
from .cvat_manager import CVATManager

__all__ = ["CarSegmenter", "CVATManager", "DEFAULT_CATEGORIES", "DEFAULT_VEHICLE_CLASSES", "CLASS_COLORS"]
