import time
import os
import csv
import gym
import slimevolleygym
from gym import wrappers
from utils import plot_learning_curve

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

def openCsv_to_plot(dir):
    score_history = []  
    with open(dir+"/progress.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            score_history.append(float(row['rollout/ep_rew_mean']))

    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, dir+"/plot.png")
pass

if __name__ == "__main__":

    ITERATIONS = 1000 
    TIME = time.strftime ('%Y_%m_%d_%H_%S')
    LOGDIR = "tmp/"+str(ITERATIONS)+"_Slime-v1_"+ TIME

    newlog = configure(folder=LOGDIR)
    
    env_wrap = gym.make("SlimeVolley-v0")
    #env_wrap = make_vec_env("SlimeVolley-v0",n_envs=1)
    model = PPO("MlpPolicy", env_wrap, verbose=1)
    #model.set_logger(newlog)

    time_steps = ITERATIONS*2048

    #model.learn(total_timesteps=time_steps)
    #model.save(os.path.join(LOGDIR, "final_model")) 

    test = "tmp/1000_Slime-v1_"+"2021_12_04_11_35"
    model = model.load(test+"/final_model")

    #openCsv_to_plot(LOGDIR)
    env = wrappers.Monitor(env_wrap, './videos/Slime-v1_'
                       + time.strftime ('%Y_%m_%d_%H_%S')
                       +'/', force = True)
    obs = env.reset()
    #while True:
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

pass