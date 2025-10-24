# risk_detection/utils/geometry_utils.py
import numpy as np
from shapely.geometry import Polygon, Point, LineString, box as shapely_box

def xyxy_to_polygon(xyxy):
    x1, y1, x2, y2 = xyxy
    return shapely_box(x1, y1, x2, y2)

def boxes_to_polys_by_name(det_obj, names_filter):
    """det_obj: sv.Detections con .xyxy y det_obj.data['class_name'] np.ndarray"""
    cls = det_obj.data.get("class_name", None)
    if cls is None:
        return {}
    mask = np.isin(cls, names_filter)
    polys = {}
    for name, box in zip(cls[mask], det_obj.xyxy[mask]):
        polys[name] = xyxy_to_polygon(box)
    return polys

def has_all_classes(det_obj, required):
    cls = det_obj.data.get("class_name", None)
    if cls is None:
        return False
    return set(required).issubset(set(cls))

def point_in_or_touch_poly(point_xy, poly: Polygon):
    p = Point(float(point_xy[0]), float(point_xy[1]))
    return p.within(poly) or p.touches(poly)

def feet_distance_to_geom(feet_points, geom, thresh):
    for x, y in feet_points:
        if Point(x, y).distance(geom) <= thresh:
            return True
    return False

def make_line_from_stickout_to_llavetm(stickout_xyxy, llave_xyxy):
    x1, y1, x2, y2 = stickout_xyxy
    xL1, yL1, xL2, yL2 = llave_xyxy
    p1 = (x1, y2)                         # esquina inferior izquierda stickout
    p2 = ((xL1 + xL2) / 2.0, yL2)         # centro inferior llavetm120
    return LineString([p1, p2]), p1, p2
