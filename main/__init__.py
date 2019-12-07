import gym
import numpy as np

from decorators.record import record
from main.action import Action
from main.cart_pole import run_episode

num_of_episodes = 350
env = gym.make('CartPole-v1')


@record(env=env, enabled=False)
def main():
    q_table = np.zeros(((1, 1, 6, 12) + (env.action_space.n,)))
    scores = np.zeros((num_of_episodes, 1))
    for i in range(num_of_episodes):
        q_table, total_reward = run_episode(env, q_table, i)
        scores[i] = total_reward
        if np.average(scores[-100:] >= 195):
            print(f"Converged in {i + 1} episodes!")
            break
    env.close()


if __name__ == "__main__":
    main()
