import math
import numpy as np


def get_pos(vehicle):
    trans = vehicle.get_transform()
    x = trans.location.x
    y = trans.location.y
    return x, y


def get_preview_lane_dis(waypoints, x, y, idx=2):
    waypt = waypoints[idx]
    vec = np.array([x - waypt[0], y - waypt[1]])
    lv = np.linalg.norm(np.array(vec))
    w = np.array([np.cos(waypt[2] / 180 * np.pi), np.sin(waypt[2] / 180 * np.pi)])
    cross = np.cross(w, vec / lv)
    dis = - lv * cross
    return dis, w


def is_within_distance_ahead(target_location, current_location, orientation, max_distance):
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)
    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

    return d_angle < 90.0


def compute_magnitude_angle(target_location, current_location, orientation):
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

    return (norm_target, d_angle)


def distance_vehicle(waypoint, vehicle_transform):
    loc = vehicle_transform.location
    dx = waypoint.transform.location.x - loc.x
    dy = waypoint.transform.location.y - loc.y

    return math.sqrt(dx * dx + dy * dy)
