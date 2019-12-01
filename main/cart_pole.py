import numpy as np


def discretize_state(observation):
    pole_angle = range(-6, 7)
    pole_velocity = range(0, 13)
    mapped_pole_angle_index = np.digitize(observation[2], pole_angle)
    mapped_pole_velocity_index = np.digitize(observation[3], pole_velocity)
    return np.array([0, 0, pole_angle[mapped_pole_angle_index], pole_velocity[mapped_pole_velocity_index]])


def run_episode(env, q_table):
    env.reset()
    done = False
    while not done:
        env.render()
        observation, reward, done, _ = env.step(0)
        observation = discretize_state(observation)
    env.close()
