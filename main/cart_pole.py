import random

import numpy as np

from main import Action

alpha = .1
discount_factor = .2


def discretize_state(observation):
    pole_angle = range(-6, 7)
    pole_velocity = range(0, 13)
    mapped_pole_angle_index = np.digitize(observation[2], pole_angle)
    mapped_pole_velocity_index = np.digitize(observation[3], pole_velocity)
    return np.array([0, 0, pole_angle[mapped_pole_angle_index], pole_velocity[mapped_pole_velocity_index]])


def random_action():
    return random.choice(list(Action))


def update_q_table(q_table, old_observation, observation, action, reward):
    old_state = q_table[tuple(old_observation)][action.value]
    new_state = q_table[tuple(observation)]
    q_table[tuple(old_observation)][action.value] += alpha * (reward + discount_factor * np.max(new_state)) - old_state
    return q_table


def run_episode(env, q_table):
    env.reset()
    done = False
    old_observation = discretize_state(env.state)
    while not done:
        env.render()
        action = random_action()
        observation, reward, done, _ = env.step(action.value)
        observation = discretize_state(observation)
        q_table = update_q_table(q_table, old_observation, observation, action, reward)
        old_observation = observation
        print('x')
    env.close()
    return q_table
