import gym
import numpy as np

from main.action import Action
from main.cart_pole import run_episode


def main():
    env = gym.make('CartPole-v1')
    q_table = np.zeros(((1, 1, 12, 12,) + (env.action_space.n,)))
    run_episode(env, q_table)


if __name__ == "__main__":
    main()
