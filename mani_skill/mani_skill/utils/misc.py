import numpy as np


def sample_from_tuple_or_scalar(rng, x):
    if isinstance(x, tuple):
        return rng.uniform(low=x[0], high=x[1])
    else:
        return x

import pathlib, yaml
def get_model_ids_from_yaml(yaml_file_path):
    path = pathlib.Path(yaml_file_path).resolve()
    with path.open("r") as f:
        raw_yaml = yaml.load(f, Loader=yaml.SafeLoader)
    return list(raw_yaml.keys())

def get_raw_yaml(yaml_file_path):
    path = pathlib.Path(yaml_file_path).resolve()
    with path.open("r") as f:
        raw_yaml = yaml.load(f, Loader=yaml.SafeLoader)
    return raw_yaml



def get_actor_state(actor):
    '''
    returns actor state with shape (13, )
    actor_state[:3] = pose p
    actor_state[3:7] = pose q
    actor_state[7:10] = velocity
    actor_state[10:13] = angular velocity
    '''
    pose = actor.get_pose()

    p = pose.p # (3, )
    q = pose.q # (4, )
    vel = actor.get_velocity() # (3, )
    ang_vel = actor.get_angular_velocity() # (3, )

    return np.concatenate([p, q, vel, ang_vel], axis=0)

def get_articulation_state(art):
    root_link = art.get_links()[0]
    base_pose = root_link.get_pose()
    base_vel = root_link.get_velocity()
    base_ang_vel = root_link.get_angular_velocity()
    qpos = art.get_qpos()
    qvel = art.get_qvel()
    return base_pose.p, base_pose.q, base_vel, base_ang_vel, qpos, qvel

def get_pad_articulation_state(art, max_dof):
    base_pos, base_quat, base_vel, base_ang_vel, qpos, qvel = get_articulation_state(art)
    k = len(qpos)
    pad_obj_internal_state = np.zeros(2 * max_dof)
    pad_obj_internal_state[:k] = qpos
    pad_obj_internal_state[max_dof : max_dof+k] = qvel
    return np.concatenate([base_pos, base_quat, base_vel, base_ang_vel, pad_obj_internal_state])


def compute_generalized_external_force(scene, robot):
    time_step = scene.get_timestep()
    link_set = set(robot.get_links())
    num_links = len(link_set)
    link_force = np.zeros((num_links, 3), dtype=float)
    link_torque = np.zeros((num_links, 3), dtype=float)
    contacts = scene.get_contacts()
    for contact in contacts:
        names = {contact.actor0.name, contact.actor1.name}
        actors = {contact.actor0, contact.actor1}
        if actors.issubset(link_set) or not actors.intersection(link_set):
            continue
        total_impulse = np.sum([point.impulse for point in contact.points], axis=0)
        if np.linalg.norm(total_impulse) < 1e-6:
            continue

        # for link_idx, link in enumerate(robot.get_links()):
            # if link in actors:
        for link in actors:
            if link not in link_set:
                continue
            link_idx = link.get_index()
            link_origin = link.pose.p
            impulse = np.zeros(3)
            angular_impulse = np.zeros(3)
            for point in contact.points:
                impulse += point.impulse
                angular_impulse += np.cross(
                    (point.position - link_origin), point.impulse
                )
            # print(
            #     contact.actor0.name,
            #     contact.actor1.name,
            #     impulse / time_step,
            #     angular_impulse / time_step,
            # )
            # NOTE(jigu): impulse is applied on actor0
            if contact.actor0 == link:
                link_force[link_idx] += impulse / time_step
                link_torque[link_idx] += angular_impulse / time_step
            else:
                link_force[link_idx] -= impulse / time_step
                link_torque[link_idx] -= angular_impulse / time_step
            break
    F = -robot.compute_generalized_external_force(link_force, link_torque)
    return F
