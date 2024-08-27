import magnum as mn
import habitat_sim
import numpy as np

# https://github.com/facebookresearch/habitat-sim/blob/366294cadd914791e57d7d70e61ae67026386f0b/examples/tutorials/colabs/ECCV_2020_Advanced_Features.ipynb
def get_2d_point(sim, sensor_name, point_3d):
    # get the scene render camera and sensor object
    render_camera = sim._sensors[sensor_name]._sensor_object.render_camera

    # use the camera and projection matrices to transform the point onto the near plane
    projected_point_3d = render_camera.projection_matrix.transform_point(
        render_camera.camera_matrix.transform_point(point_3d)
    )

    if projected_point_3d[2] < 0:
        return None

    # convert the 3D near plane point to integer pixel space
    point_2d = mn.Vector2(projected_point_3d[0], -projected_point_3d[1])
    point_2d = point_2d / render_camera.projection_size()[0]
    point_2d += mn.Vector2(0.5)
    point_2d *= render_camera.viewport
    return mn.Vector2i(point_2d)


def is_target_occluded(
    target: mn.Vector3,
    agent_position: mn.Vector3,
    agent_height: float,
    task,
    ignore_object_ids: set = set(),
    ignore_objects: bool = False,
    ignore_receptacles: bool = False,
    ignore_non_negative: bool = False,
) -> bool:

    sim = task._sim
    ray = habitat_sim.geo.Ray()
    ray.origin = agent_position + mn.Vector3(0, agent_height, 0)
    ray.direction = target - ray.origin
    raycast_results = sim.cast_ray(ray)

    hits = [
        x.object_id
        for x in raycast_results.hits
        if x.ray_distance < 1 and x.object_id not in ignore_object_ids
    ]
    if ignore_objects:
        hits = [x for x in hits if x not in task.all_object_ids]
    if ignore_receptacles:
        hits = [x for x in hits if x not in task.all_receptacles_ids]
    if ignore_non_negative:
        hits = [x for x in hits if x < 0]

    if hits:
        return True
    return False


def is_connected(l2_dist, dist, angle_diff, threshold=0.1):
    # print(angle_diff)
    return np.abs(l2_dist / dist - 1) < 0.15 and np.abs(angle_diff) < 20 / 180 * np.pi


def get_dists(c, points, task):
    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = c
    path.requested_ends = points
    did_find_a_path = task._sim.pathfinder.find_path(path)
    if not did_find_a_path:
        return None
    dist = path.geodesic_distance
    l2_dist = np.linalg.norm(points[path.closest_end_point_index] - c)
    return l2_dist, dist, path.closest_end_point_index


def get_angle(a, b):
    return np.arctan2(*np.asarray(b - a)[[2, 0]])


def remove_closest(c, points, task, angle=None):
    points = sorted(points, key=lambda x: np.linalg.norm(x - c), reverse=False)
    out = get_dists(c, points, task)
    if out is None:
        return points
    l2_dist, dist, closest_end_point_index = out
    angle_new = get_angle(c, points[0])
    if angle is not None:
        angle_diff = angle_new - angle
    else:
        angle_diff = 0
    if is_connected(l2_dist, dist, angle_diff=angle_diff):
        return remove_closest(
            points[closest_end_point_index],
            [x for i, x in enumerate(points) if i != closest_end_point_index],
            angle=angle_new,
            task=task,
        )
    else:
        return points


def process_navigation_points(c, points, task):
    new_points = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if points[j] == points[i]:
                break
        else:
            new_points.append(points[i])
    points = new_points

    to_keep = []
    while points:
        last_len = len(points)
        points = sorted(points, key=lambda x: np.linalg.norm(x - c), reverse=False)
        to_keep.append(points[0])
        pot_points = []
        last_len = -1
        while len(pot_points) != last_len:
            last_len = len(pot_points)
            pot_points = remove_closest(points[0], points[1:], task=task)
            points = [points[0], *pot_points]
        points = points[1:]
    return to_keep


def get_navigation_points(
    c,
    task,
    r=2,
    n=60,
    eps=0.1,
    angle_tol=5,
    max_num_trials=20,
    height=1.2,
    target_object_id=None,
    ignore_objects: bool = False,
    ignore_receptacles: bool = False,
    ignore_non_negative: bool = False,
    cleanup_nav_points: bool = False,
):
    c = np.asarray(c)
    output = []
    for i in np.linspace(0, 2 * np.pi, n):
        for trial_i in range(1, max_num_trials + 1):
            shift = [(trial_i * eps) * np.cos(i), 0, (trial_i * eps) * np.sin(i)]
            projection = task._sim.pathfinder.snap_point(
                c + shift, task._sim.largest_island_idx
            )
            dst = np.linalg.norm(projection - c)
            if (
                dst <= r
                and np.abs(np.arctan2(*np.asarray(projection - c)[[2, 0]]) - i)
                < angle_tol / 180 * np.pi
            ):
                break

        if dst <= r and not is_target_occluded(
            c,
            projection,
            height,
            task,
            ignore_object_ids=[target_object_id],
            ignore_objects=ignore_objects,
            ignore_receptacles=ignore_receptacles,
            ignore_non_negative=ignore_non_negative,
        ):
            output.append(projection)

    if cleanup_nav_points:
        output = process_navigation_points(c, output, task)
    return output


def get_navigation_points_grid(
    c,
    task,
    r=2,
    n=60,
    eps=0.1,
    angle_tol=5,
    max_num_trials=20,
    height=1.2,
    target_object_id=None,
    ignore_objects: bool = False,
    ignore_receptacles: bool = False,
    ignore_non_negative: bool = False,
    cleanup_nav_points: bool = False,
):
    for k in range(max_num_trials):
        c = np.asarray(c)
        island_y = task._sim.pathfinder.get_random_navigable_point_near(
            c, 3, island_index=task._sim.largest_island_idx
        )[1]
        points = np.meshgrid(np.arange(-r, r, eps), np.arange(-r, r, eps))
        mask = points[0] ** 2 + points[1] ** 2 < r**2 - eps
        xs = points[0][mask] + c[0]
        ys = np.zeros_like(xs) + island_y
        zs = points[1][mask] + c[2]
        filtered_points = [
            (x, y, z)
            for x, y, z in zip(xs, ys, zs)
            if task._sim.is_navigable([x, y, z])
        ]
        output = [
            np.asarray(x)
            for x in filtered_points
            if not is_target_occluded(
                c,
                np.asarray(x),
                height + eps * k,
                task,
                ignore_object_ids=[target_object_id],
                ignore_objects=ignore_objects,
                ignore_receptacles=ignore_receptacles,
                ignore_non_negative=ignore_non_negative,
            )
        ]
        if cleanup_nav_points:
            output = process_navigation_points(c, output, task)
        if output:
            break
    return output


def get_obj_pixel_counts(task, margin=None, strict=True):
    sim_obs = task._sim.get_sensor_observations()
    observations = task._sim._sensor_suite.sensors["head_panoptic"].get_observation(
        sim_obs
    )
    objs_ids, objs_count = np.unique(observations, return_counts=True)
    objs_ids -= task._sim.habitat_config.object_ids_start
    id2count = dict(zip(objs_ids, objs_count))
    if margin is not None:
        observations[margin:-margin, margin:-margin] = -10000
        margin_objs_ids, margin_objs_count = np.unique(observations, return_counts=True)
        margin_objs_ids -= task._sim.habitat_config.object_ids_start
        margin_id2count = dict(zip(margin_objs_ids, margin_objs_count))
        for k in margin_id2count:
            if k in id2count and margin_id2count[k] > (0 if strict else id2count[k]):
                del id2count[k]
    return id2count
