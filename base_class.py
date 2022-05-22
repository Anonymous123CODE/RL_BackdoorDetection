import time
from abc import ABC, abstractmethod
from collections import deque
import os
import io
import zipfile

import gym
import tensorflow as tf
import numpy as np

import logger
from common_policies import get_policy_from_name
from utils import set_random_seed, get_schedule_fn






class BaseRLModel(ABC):
    """
    The base RL model

    :param policy: (BasePolicy) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param policy_base: (BasePolicy) the base policy used by this method
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 debug
    :param support_multi_env: (bool) Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: (bool) When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: (int) Seed for the pseudo random generators
    """
    def __init__(self, policy, env, policy_base, policy_kwargs=None,
                 verbose=0, device='auto', support_multi_env=False,
                 create_eval_env=False, monitor_wrapper=True, seed=None):
        if isinstance(policy, str) and policy_base is not None:
            self.policy_class = get_policy_from_name(policy_base, policy)
        else:
            self.policy_class = policy

        self.env = env
        # get VecNormalize object if needed
        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.observation_space = None
        self.action_space = None

        self.num_timesteps = 0
        self.eval_env = None
        self.replay_buffer = None
        self.seed = seed
        self.action_noise = None

        # Track the training progress (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress = 1

        # Create and wrap the env if needed

    def _setup_learning_rate(self):
        """Transform to callable if needed."""
        self.learning_rate = get_schedule_fn(self.learning_rate)



    def _update_learning_rate(self, optimizers):
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress (from 1 to 0).

        :param optimizers: ([th.optim.Optimizer] or Optimizer) An optimizer
            or a list of optimizer.
        """
        # Log the current learning rate
        logger.logkv("learning_rate", self.learning_rate(self._current_progress))

        # if not isinstance(optimizers, list):
        #     optimizers = [optimizers]
        # for optimizer in optimizers:
        #     update_learning_rate(optimizer, self.learning_rate(self._current_progress))

    @staticmethod
    def safe_mean(arr):
        """
        Compute the mean of an array if there is at least one element.
        For empty array, return nan. It is used for logging only.

        :param arr: (np.ndarray)
        :return: (float)
        """
        return np.nan if len(arr) == 0 else np.mean(arr)

    def get_env(self):
        """
        returns the current environment (can be None if not defined)

        :return: (gym.Env) The current environment
        """
        return self.env

    def set_env(self, env):
        """
        :param env: (gym.Env) The environment for learning a policy
        """
        raise NotImplementedError()

    @abstractmethod
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="run",
              eval_env=None, eval_freq=-1, n_eval_episodes=5, reset_num_timesteps=True):
        """
        Return a trained model.

        :param total_timesteps: (int) The total number of samples to train on
        :param callback: (function (dict, dict)) -> boolean function called at every steps with state of the algorithm.
            It takes the local and global variables. If it returns False, training is aborted.
        :param log_interval: (int) The number of timesteps before logging.
        :param tb_log_name: (str) the name of the run for tensorboard log
        :param reset_num_timesteps: (bool) whether or not to reset the current timestep number (used in logging)
        :param eval_env: (gym.Env) Environment that will be used to evaluate the agent
        :param eval_freq: (int) Evaluate the agent every `eval_freq` timesteps (this may vary a little)
        :param n_eval_episodes: (int) Number of episode to evaluate the agent
        :return: (BaseRLModel) the trained model
        """
        pass

    @abstractmethod
    def predict(self, observation, state=None, mask=None, deterministic=False):
        """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        pass

    def set_random_seed(self, seed=None):
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed: (int)
        """
        if seed is None:
            return
        set_random_seed(seed)
        # self.action_space.seed(seed)
        if self.env is not None:
            self.env.seed(seed)
        if self.eval_env is not None:
            self.eval_env.seed(seed)

    # def _setup_learn(self, eval_env):
    #     """
    #     Initialize different variables needed for training.
    #
    #     :param eval_env: (gym.Env or VecEnv)
    #     :return: (int, int, [float], np.ndarray, VecEnv)
    #     """
    #     self.start_time = time.time()
    #     self.ep_info_buffer = deque(maxlen=100)
    #
    #     if self.action_noise is not None:
    #         self.action_noise.reset()
    #
    #     timesteps_since_eval, episode_num = 0, 0
    #     evaluations = []
    #
    #     if eval_env is not None and self.seed is not None:
    #         eval_env.seed(self.seed)
    #
    #
    #     obs = self.env.reset()
    #     return timesteps_since_eval, episode_num, evaluations, obs, eval_env

    # def _update_info_buffer(self, infos):
    #     """
    #     Retrieve reward and episode length and update the buffer
    #     if using Monitor wrapper.
    #
    #     :param infos: ([dict])
    #     """
    #     for info in infos:
    #         maybe_ep_info = info.get('episode')
    #         if maybe_ep_info is not None:
    #             self.ep_info_buffer.extend([maybe_ep_info])
    #
    # def _eval_policy(self, eval_freq, eval_env, n_eval_episodes,
    #                  timesteps_since_eval, deterministic=True):
    #     """
    #     Evaluate the current policy on a test environment.
    #
    #     :param eval_env: (gym.Env) Environment that will be used to evaluate the agent
    #     :param eval_freq: (int) Evaluate the agent every `eval_freq` timesteps (this may vary a little)
    #     :param n_eval_episodes: (int) Number of episode to evaluate the agent
    #     :parma timesteps_since_eval: (int) Number of timesteps since last evaluation
    #     :param deterministic: (bool) Whether to use deterministic or stochastic actions
    #     :return: (int) Number of timesteps since last evaluation
    #     """
    #     if 0 < eval_freq <= timesteps_since_eval and eval_env is not None:
    #         timesteps_since_eval %= eval_freq
    #         # Synchronise the normalization stats if needed
    #         sync_envs_normalization(self.env, eval_env)
    #         mean_reward, std_reward = evaluate_policy(self, eval_env, n_eval_episodes, deterministic=deterministic)
    #         if self.verbose > 0:
    #             print("Eval num_timesteps={}, "
    #                   "episode_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward, std_reward))
    #             print("FPS: {:.2f}".format(self.num_timesteps / (time.time() - self.start_time)))
    #     return timesteps_since_eval
