import math
import random

import numpy as np

from main import Action

discount_factor = 1


def get_alpha(index):
    return max(.1, min(1, 1.0 - math.log10((index + 1) / 25)))


def get_epsilon(index):
    return max(.1, min(1, 1.0 - math.log10((index + 1) / 25)))


def discretize_state(observation, observation_space):
    higher_bounds = np.array([observation_space.high[0], .5, observation_space.high[2], math.radians(50)])
    lower_bounds = np.array([observation_space.low[0], -.5, observation_space.low[2], -math.radians(50)])
    new_higher_bounds = np.array([1, 1, 6, 12])
    scale_ratios = observation + np.abs(lower_bounds) / (higher_bounds - lower_bounds)
    new_observation = np.round((new_higher_bounds - 1) * scale_ratios).astype(int)
    return np.minimum(new_higher_bounds - 1, np.maximum(0, new_observation))


def random_action():
    return random.choice(list(Action))


def update_q_table(q_table, old_observation, observation, action, reward, index):
    old_state = q_table[tuple(old_observation)][action.value]
    new_state = q_table[tuple(observation)]
    q_table[tuple(old_observation)][action.value] += get_alpha(index) * (
        reward + discount_factor * np.max(new_state) - old_state)
    return q_table


def select_action(q_table, old_observation, index):
    epsilon = get_epsilon(index)
    if random.uniform(0, 1) < epsilon:
        return random_action()
    else:
        state = q_table[tuple(old_observation)]
        action_idx = np.argmax(state)
        return Action(int(action_idx))


def run_episode(env, q_table, index):
    env.reset()
    done = False
    old_observation = discretize_state(env.state, env.observation_space)
    total_reward = 0
    while not done:
        env.render()
        action = select_action(q_table, old_observation, index)
        observation, reward, done, _ = env.step(action.value)
        observation = discretize_state(observation, env.observation_space)
        q_table = update_q_table(q_table, old_observation, observation, action, reward, index)
        old_observation = observation
        total_reward += reward
        if total_reward >= 195:
            print(f"Converged in {index + 1} episodes!")
            break
    return q_table
