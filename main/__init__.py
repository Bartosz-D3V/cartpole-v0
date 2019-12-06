import gym
import numpy as np

from main.action import Action
from main.cart_pole import run_episode


def main():
    num_of_episodes = 210
    env = gym.make('CartPole-v1')
    q_table = np.zeros(((1, 1, 6, 12) + (env.action_space.n,)))
    for i in range(num_of_episodes):
        q_table = run_episode(env, q_table, i)
    env.close()


if __name__ == "__main__":
    main()
