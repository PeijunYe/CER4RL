import random
import time
import carla
import numpy as np
from compute_function import get_preview_lane_dis, get_pos
from route_planner import RoutePlanner


class CarEnv:
    number_of_vehicles = 50,
    number_of_walkers = 50,
    max_past_step = 1
    discrete = False,
    discrete_acc = [-3.0, 0.0, 3.0],
    discrete_steer = [-0.2, 0.0, 0.2],
    continuous_accel_range = [-3.0, 3.0],
    continuous_steer_range = [-0.3, 0.3],
    ego_vehicle_filter = 'vehicle.lincoln*',
    port = 2000,
    town = 'Town03',
    max_time_episode = 1000,
    max_waypt = 12,
    obs_range = 32,
    lidar_bin = 0.125,
    d_behind = 12,
    out_lane_thres = 2.0,
    desired_speed = 8,
    max_ego_spawn_times = 200,
    display_route = True,
    pixor_size = 64,
    pixor = False,
    dests = [[4.46, -61.46, 0], [-49.53, -2.89, 0], [-6.48, 55.47, 0], [35.96, 3.33, 0]]
    STEER_AMT = 1.0
    SECONDS_PER_EPISODE = 10.0
    REPLAY_MEMORY_SIZE = 5000
    MIN_REPLAY_MEMORY_SIZE = 1000
    MINIBATCH_SIZE = 16
    PREDICTION_BATCH_SIZE = 1
    TRAINNING_BATCH_SIZE = MINIBATCH_SIZE // 4
    UPDATE_TARGET_EVERY = 5
    MEMORY_FRACTION = 0.8

    def __init__(self):
        number_of_walkers = 50
        ego_vehicle_filter = 'vehicle.lincoln*'

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        self.walker_spawn_points = []
        for i in range(number_of_walkers):
            spawn_point = carla.Transform()
            spawn_point.location = self.world.get_random_location_from_navigation()
            if (spawn_point.location != None):
                spawn_point.location = spawn_point.location
                self.walker_spawn_points.append(spawn_point)

        self.ego_bp = self._create_vehicle_bluepprint(ego_vehicle_filter, color='49,8,8')

        self.collision_hist = []
        self.collision_hist_l = 1
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = 0.1

        self.reset_step = 0
        self.total_step = 0
        self.synchronous_master = True

    def reset(self):

        self.collision_sensor = None

        self._clear_all_actors(['sensor.other.collision', 'vehicle.*', 'controller.ai.walker', 'walker.*'])

        self._set_synchronous_mode(False)

        random.shuffle(self.vehicle_spawn_points)
        count = self.number_of_vehicles[0]
        if count > 0:
            for spawn_point in self.vehicle_spawn_points:
                if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
                count -= 1

        random.shuffle(self.walker_spawn_points)
        count = self.number_of_walkers[0]
        if count > 0:
            for spawn_point in self.walker_spawn_points:
                if self._try_spawn_random_walker_at(spawn_point):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
                count -= 1

        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        self.walker_polygons = []
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)

        ego_spawn_times = 0
        while True:
            if ego_spawn_times > self.max_ego_spawn_times[0]:
                self.reset()

            transform = random.choice(self.vehicle_spawn_points)

            if self._try_spawn_ego_vehicle_at(transform):
                break
            else:
                ego_spawn_times += 1
                time.sleep(0.1)

        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)

        self.collision_hist = []

        self.time_step = 0
        self.reset_step += 1

        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        return self._get_obs()

    def step(self, action):
        acc = action[0]
        steer = action[1]

        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 8, 0, 1)

        act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
        self.ego.apply_control(act)
        self.world.tick()

        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:
            self.vehicle_polygons.pop(0)
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)
        while len(self.walker_polygons) > self.max_past_step:
            self.walker_polygons.pop(0)

        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        info = {
            'waypoints': self.waypoints,
            'vehicle_front': self.vehicle_front
        }

        self.time_step += 1
        self.total_step += 1

        return (self._get_obs(), self._get_reward(self._get_obs()), self._terminal(), info)

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if
                                                     int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _clear_all_actors(self, actor_filters):
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker':
                        actor.stop()
                    actor.destroy()

    def _set_synchronous_mode(self, synchronous=True):
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot()
            return True
        return False

    def _try_spawn_random_walker_at(self, transform):
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)

        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
            walker_controller_actor.start()
            walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
            walker_controller_actor.set_max_speed(1 + random.random())
            return True
        return False

    def _get_actor_polygons(self, filt):
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _try_spawn_ego_vehicle_at(self, transform):
        vehicle = None
        overlap = False
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break

        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            self.ego = vehicle
            return True

        return False

    def _get_obs(self):
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        delta_yaw = np.arcsin(np.cross(w,
                                       np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x ** 2 + v.y ** 2)

        ego_o = self.ego.get_angular_velocity()
        ego_angular = np.sqrt(ego_o.x ** 2 + ego_o.y ** 2)

        actor_list = self.world.get_actors()
        light_state = 0
        if self.ego.is_at_traffic_light():
            light_state = 1
        light_distances = []
        lights_lists = actor_list.filter("*traffic_light*")
        for lights_list in lights_lists:
            transform_light = lights_list.get_transform()
            transform_light_location = transform_light.location
            light_location = carla.Location(transform_light_location)
            ego_location = ego_trans.location
            dis = light_location.distance(ego_location)
            light_distances.append(dis)

        other_vehicle_list = []
        ohter_vehicles = actor_list.filter("*vehicle*")
        for other_vehicle in ohter_vehicles:
            transform_vehicle = other_vehicle.get_transform()
            transform_vehicle_location = transform_vehicle.location
            vehicle_location = carla.Location(transform_vehicle_location)
            ego_location = ego_trans.location
            box = self.ego.bounding_box
            limit_dis = box.location.distance(ego_location)
            dis = vehicle_location.distance(ego_location)

            if dis < 0.5 * limit_dis:
                v = other_vehicle.get_velocity()
                other_vehicle_speed = np.sqrt(v.x ** 2 + v.y ** 2)
                a = other_vehicle.get_acceleration()
                other_vehicle_acc = np.sqrt(a.x ** 2 + a.y ** 2)
                o = other_vehicle.get_angular_velocity()
                other_vehicle_angular = np.sqrt(o.x ** 2 + o.y ** 2)
                other_vehicle_list.append(other_vehicle_speed)
                other_vehicle_list.append(other_vehicle_acc)
                other_vehicle_list.append(other_vehicle_angular)
                other_vehicle_list.append(dis)

        if len(other_vehicle_list) > 36:
            other_vehicle_list = [0 for _ in range(36)]
        else:
            while len(other_vehicle_list) < 36:
                other_vehicle_list.append(0)

        if len(self.collision_hist) > 0:
            done = 1
        else:
            done = 0

        ego_x, ego_y = get_pos(self.ego)

        if self.time_step > self.max_time_episode[0]:
            mark = 1
        else:
            mark = 0

        goal = 0
        if self.dests is not None:
            for dest in self.dests:
                if np.sqrt((ego_x - dest[0]) ** 2 + (ego_y - dest[1]) ** 2) < 4:
                    goal = 1
                    break

        state = [done, mark, goal, lateral_dis, - delta_yaw, speed, self.vehicle_front, light_state, ego_angular]
        sum_state = state + other_vehicle_list

        state = np.array(sum_state)

        obs = state

        return obs

    def _get_reward(self, state):
        r1 = -1 * state[0]
        r2 = state[1]
        r3 = 5 * state[2]

        return r1 + r2 + r3

    def _terminal(self):
        ego_x, ego_y = get_pos(self.ego)

        if len(self.collision_hist) > 0:
            return True

        if self.time_step > self.max_time_episode[0]:
            return True

        if self.dests is not None:
            for dest in self.dests:
                if np.sqrt((ego_x - dest[0]) ** 2 + (ego_y - dest[1]) ** 2) < 4:
                    return True

        return False
