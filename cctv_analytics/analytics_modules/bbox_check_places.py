import json
from shapely.geometry import Polygon, box
from typing import Optional, List
from utils.config import Place


def check_bbox_overlap(bbox, places: List[Place]):
    """Check which polygons the bounding box overlaps with."""
    x1, y1, x2, y2 = bbox
    bbox_poly = box(x1, y1, x2, y2)  # Create a Shapely box (rectangle)

    overlapping_polygons = []
    for place in places:
        poly = Polygon(place.polygon)  # Create a Shapely polygon

        if bbox_poly.intersects(poly):
            overlapping_polygons.append(place.name)

    return overlapping_polygons


def check_place(bbox, places: List[Place]):
    overlapping_places = check_bbox_overlap(bbox, places)

    if overlapping_places:
        return overlapping_places
    else:
        return []
