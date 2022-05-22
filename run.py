import gym
import gym_compete
from ppo import  PPO
import tensorflow.keras as keras
from policies import MlpPolicy
import os 
import tensorflow as tf
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--terminal', type=bool,
                    choices=[True, False],
                    default=False)
parser.add_argument('--cuda', type=str)
parser.add_argument('--seed', type=int)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
env = gym.make("multicomp/RunToGoalHumans-v0")

model_name = "saved_models/human-to-go/trojan_model_128.h5"
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])
oppo_model = keras.models.load_model(model_name)
seed = np.random.randint(1e5)
model = PPO(MlpPolicy, env,
            verbose=1,
            oppo_agent=oppo_model, seed = seed)
model.learn(max_epochs=10000)
