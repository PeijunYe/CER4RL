from enum import Enum
from collections import deque

import carla

from compute_function import distance_vehicle, is_within_distance_ahead, compute_magnitude_angle


class RoadOption(Enum):
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4


class RoutePlanner():
    def __init__(self, vehicle, buffer_size):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()

        self._sampling_radius = 5
        self._min_distance = 4

        self._target_waypoint = None
        self._buffer_size = buffer_size
        self._waypoint_buffer = deque(maxlen=self._buffer_size[0])

        self._waypoints_queue = deque(maxlen=600)
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))
        self._target_road_option = RoadOption.LANEFOLLOW

        self._last_traffic_light = None
        self._proximity_threshold = 15.0

        self._compute_next_waypoints(k=200)

    def _compute_next_waypoints(self, k=1):
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 1:
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                road_options_list = retrieve_options(
                    next_waypoints, last_waypoint)

                road_option = road_options_list[1]

                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]

            self._waypoints_queue.append((next_waypoint, road_option))

    def run_step(self):
        waypoints = self._get_waypoints()
        red_light, vehicle_front = self._get_hazard()
        return waypoints, red_light, vehicle_front

    def _get_waypoints(self):
        if len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            self._compute_next_waypoints(k=100)

        while len(self._waypoint_buffer) < self._buffer_size[0]:
            if self._waypoints_queue:
                self._waypoint_buffer.append(
                    self._waypoints_queue.popleft())
            else:
                break

        waypoints = []

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            waypoints.append(
                [waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.rotation.yaw])

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._target_waypoint, self._target_road_option = self._waypoint_buffer[0]

        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if distance_vehicle(
                    waypoint, vehicle_transform) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index - 1):
                self._waypoint_buffer.popleft()

        return waypoints

    def _get_hazard(self):
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        vehicle_state = self._is_vehicle_hazard(vehicle_list)

        light_state = self._is_light_red_us_style(lights_list)

        return light_state, vehicle_state

    def _is_vehicle_hazard(self, vehicle_list):

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            loc = target_vehicle.get_location()
            if is_within_distance_ahead(loc, ego_vehicle_location,
                                        self._vehicle.get_transform().rotation.yaw,
                                        self._proximity_threshold):
                return True

        return False

    def _is_light_red_us_style(self, lights_list):
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        if ego_vehicle_waypoint.is_intersection:
            return False

        if self._target_waypoint is not None:
            if self._target_waypoint.is_intersection:
                min_angle = 180.0
                sel_traffic_light = None
                for traffic_light in lights_list:
                    loc = traffic_light.get_location()
                    magnitude, angle = compute_magnitude_angle(loc,
                                                               ego_vehicle_location,
                                                               self._vehicle.get_transform().rotation.yaw)
                    if magnitude < 80.0 and angle < min(25.0, min_angle):
                        sel_traffic_light = traffic_light
                        min_angle = angle

                if sel_traffic_light is not None:
                    if self._last_traffic_light is None:
                        self._last_traffic_light = sel_traffic_light

                    if self._last_traffic_light.state == carla.libcarla.TrafficLightState.Red:
                        return True
                else:
                    self._last_traffic_light = None

        return False


def retrieve_options(list_waypoints, current_waypoint):
    options = []
    for next_waypoint in list_waypoints:
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def compute_connection(current_waypoint, next_waypoint):
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < 1.0:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT
