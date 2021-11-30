import gym
import slimevolleygym
import numpy as np

from PPO import Agent

if __name__ == "__main__":
    env = gym.make('SlimeVolley-v0')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    switcher = {
        0: [0,0,0],
        1: [1,0,0],
        2: [0,1,0],
        3: [0,0,1]
    }

    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)
    obs = env.reset()
    action, prob, val = agent.choose_action(obs)
    actual_action = switcher[action]
    print(actual_action)
    """
    for i in range(10):
        obs = env.reset()
        action, prob, val = agent.choose_action(obs)
        observation_, reward, done, info = env.step([0,0,0])
        print(action)
    """
    pass