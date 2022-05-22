import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym
import gym_compete
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class play_game:

    def __init__(self, env,  model):


        self.env = env

        self.model = model
        self.ob_mean = np.load("/home/junfeng/backdoorRL/multiagent-competition/parameters/ants_to_go/obrs_mean.npy")
        self.ob_std = np.load("/home/junfeng/backdoorRL/multiagent-competition/parameters/ants_to_go/obrs_std.npy")
        np.random.seed(666)

    def play(self, n_steps=40, seed = 22,perturb=0.0,actions=None):

        self.env.seed(seed)
        obs = self.env.reset()
        #
        # if len:
        #
        #     actions = actions
        #
        # else:
        #     print("a")
        #     actions = np.array([[0.38642034, -0.77159446, -0.3736974, -0.49985278,
        #                          -0.60995424, 0.6847563, 0.9057063, 0.6158565]])


        total_reward = 0.0

        for i in range(100):

            obzs = [np.clip((obs[i] - self.ob_mean) / self.ob_std, -5.0, 5.0)
                    for i in range(len(obs))]

            action_v = np.clip(self.model.predict(np.reshape(obzs[0],(1,122,1))
                                                  ),-1,1)

            if i< n_steps:
                action_s = np.clip(

                        np.array(actions)*(1+perturb),
                                   -1,1)

            else:
                action_s = np.zeros(8)

            new_obs, rewards, dones, infos= self.env.step(tuple([action_v[0],action_s]))

            total_reward+=-rewards[0]

            obs = new_obs

        return total_reward

    def multiple_player(self,N_actions):

        reward_set = []
        for i in range(len(N_actions)):

            reward_set.append(self.play(n_steps=40, seed = 22,perturb=0.0,actions=N_actions[i]))

        return np.array(reward_set)

if __name__ == '__main__':
    per_result_4 = []
    per_result_5 = []
    env = gym.make("run-to-goal-ants-v0")
    game = play_game(env=env,model=keras.models.load_model("/home/junfeng/backdoorRL/multiagent-competition/saved_models/trojnn_incre.h5"))


    for i in range(1):
        per_result_5.append(game.play(seed=5,perturb=0.3,
                                      actions= [0.38642034, -0.77159446, -0.3736974, -0.49985278,
                               -0.60995424, 0.6847563, 0.9057063, 0.6158565]))
    #
    # for i in range(50):
    #     per_result_4.append(game.play(seed=4,perturb=0.01*i,
    #                                   actions= [0.38642034, -0.77159446, -0.3736974, -0.49985278,
    #                            -0.60995424, 0.6847563, 0.9057063, 0.6158565]))

    np.save("per_result/per4.npy", np.array(per_result_4))
    np.save("per_result/per5.npy", np.array(per_result_5))
