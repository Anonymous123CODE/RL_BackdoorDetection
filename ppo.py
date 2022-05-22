import os
import time

import gym
from gym import spaces
import tensorflow as tf
import numpy as np

from base_class import BaseRLModel
from buffers import RolloutBuffer
from utils import explained_variance
import logger
from policies import PPOPolicy


class PPO(BaseRLModel):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI spinningup (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: (PPOPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: (float or callable) The learning rate, it can be a function
        of the current progress (from 1 to 0)
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: (int) Minibatch size
    :param n_epochs: (int) Number of epoch when optimizing the surrogate loss
    :param gamma: (float) Discount factor
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: (float or callable) Clipping parameter, it can be a function of the current progress
        (from 1 to 0).
    :param clip_range_vf: (float or callable) Clipping parameter for the value function,
        it can be a function of the current progress (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param target_kl: (float) Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param seed: (int) Seed for the pseudo random generators
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

    def __init__(self, policy, env, learning_rate=3e-4,
                 n_steps=10, batch_size=32, n_epochs=10,
                 gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=10,
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                 target_kl=None,
                 policy_kwargs=None, verbose=0, seed=0,
                 _init_setup_model=True, oppo_agent=None):

        super(PPO, self).__init__(policy, env, PPOPolicy, policy_kwargs=policy_kwargs,
                                  verbose=verbose, support_multi_env=True, seed=seed)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.total_trajectories = 60
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.target_kl = target_kl
        self.oppo_agent = oppo_agent
        self.ob_mean = np.load(
            "parameters/human-to-go/obrs_mean.npy")
        self.ob_std = np.load(
            "parameters/human-to-go/obrs_std.npy")
        self.seed = seed
        if _init_setup_model:
            self._setup_model()
        self.reward_std = 184.0983

        self.reward_mean = -35.4372

        self.state_dim = 380

        self.action_dim = 17

        print(
            f"INFO:\n CLIP_RANGE:{self.clip_range};  CLIP_V_RANGE:{self.clip_range_vf}, STEPS:{self.n_steps} ,SEED:{self.seed}, TOTAL_TRojectores:{self.total_trajectories}\n")

    def normalize_reward(self, reward):

        return np.clip((reward-self.reward_mean)/self.reward_std, -10, 10)

    def _setup_model(self):
        self._setup_learning_rate()

        state_dim = 380
        action_dim = 17

        self.rollout_buffer = RolloutBuffer(buffer_size=self.n_steps, obs_dim=state_dim, action_dim=action_dim,
                                            gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=None)
        self.policy = self.policy_class(state_dim,  action_dim,
                                        self.learning_rate, **self.policy_kwargs)

        # self.clip_range = get_schedule_fn(self.clip_range)
        # if self.clip_range_vf is not None:
        #     self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def predict(self, observation, state=None, mask=None, deterministic=False):
        """
        Get the model's action from an observation

        :param   observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        clipped_actions = self.policy.actor_forward(
            np.array(observation).reshape(1, -1), deterministic=deterministic)

        clipped_actions = np.clip(clipped_actions, low=-1, high=1)
        return clipped_actions

    def collect_rollouts(self, env, rollout_buffer, n_rollout_steps=40, callback=None,
                         obs=None):

        n_steps = 0
        rollout_buffer.reset()
        reward_total = 0
        full = 0
        saved_ob = []
        saved_ac = []

        while n_steps < self.total_trajectories:

            obzs = [np.clip((obs[i] - self.ob_mean) / self.ob_std, -5.0, 5.0)
                    for i in range(len(obs))]

            if n_steps < self.n_steps:
                actions, values, log_probs = self.policy.call(
                    obzs[0].reshape((1, self.state_dim)))
                actions = actions.numpy()

            # Rescale and perform action

                clipped_actions = np.clip(actions, -1, 1)

                
            # Clip the actions to avoid out of bound error
            else:

                clipped_actions = [np.zeros(self.action_dim)]

            action_0 = self.oppo_agent.predict(
                np.reshape(obzs[1], (1, self.state_dim, 1)))

            # saved_ac.append(clipped_actions)
            # saved_ob.append(obzs[1])

            #
            new_obs, rewards, dones, infos = env.step(
                ([clipped_actions[0], action_0[0]]))
            # new_obs, rewards, dones, infos = env.step(
            #     ([action_0[0],clipped_actions[0]]))

            terminal = dones[0]

            reward_total += -rewards[1]

            if terminal:

                env.seed(self.seed)
                new_obs = env.reset()
            # self._update_info_buffer(infos)
                print(f"GG:{reward_total}")
                if reward_total>0 and n_steps > self.n_steps:
                    full = 1
                    #np.save(f"human_ob/saved_{str(self.seed)}.npy", np.array(saved_ob))
                    #np.save(f"human_ac/saved_{str(self.seed)}.npy", np.array(saved_ac))
                    break
                reward_total = 0
                
            if n_steps < self.n_steps-1:
                rollout_buffer.add(obzs[0],
                                   clipped_actions,
                                   self.normalize_reward(-rewards[1]),
                                   terminal,
                                   values,
                                   log_probs)

            if n_steps == self.n_steps-1:

                obzs_last = obzs[0]
                actions_last = clipped_actions
                value_last = values
                log_probs_last = log_probs

            n_steps += 1
            obs = new_obs

            # print(-rewards[0])
            # if only the target agent fails at the observing phase, the game will not terminate for Run-To-Goal(Humans) and the target agent will get 1000 penalty. Therefore the accumulated reward will be 
            #at least larger than 700 
            
            if reward_total >= 700 and n_steps >= self.n_steps:

                terminal = True

                rollout_buffer.add(obzs_last,
                                   actions_last,
                                   self.normalize_reward(1000),
                                   terminal,
                                   value_last,
                                   log_probs_last)
# Save Trajectories As Numpy File

                #np.save(f"human_ob/saved_{str(self.seed)}.npy", np.array(saved_ob))
                #np.save(f"human_ac/saved_{str(self.seed)}.npy", np.array(saved_ac))
                full = 1

                break

        if full == 0:

            rollout_buffer.add(obzs_last,
                               actions_last,
                               self.normalize_reward(-1000),
                               True,
                               value_last,
                               log_probs_last)

        # if reward_total >= 100:

            #self.clip_range = 0.001
        rollout_buffer.compute_returns_and_advantage(value_last, dones=True)

        print(f"Total Reward:{reward_total}")
        return full

    @tf.function
    def policy_loss(self, advantage, log_prob, old_log_prob, clip_range):
        # Normalize advantage
        advantage = (advantage - tf.reduce_mean(advantage)) / \
            (tf.math.reduce_std(advantage) + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        ratio = tf.exp(log_prob - old_log_prob)
        # clipped surrogate loss
        policy_loss_1 = advantage * ratio
        policy_loss_2 = advantage * \
            tf.clip_by_value(ratio, 1 - clip_range, 1 + clip_range)
        return - tf.reduce_mean(tf.minimum(policy_loss_1, policy_loss_2))

    @tf.function
    def value_loss(self, values, old_values, return_batch, clip_range_vf):
        if clip_range_vf is None:
            # No clipping
            values_pred = values
        else:
            # Clip the different between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = old_values + \
                tf.clip_by_value(values - old_values, -
                                 clip_range_vf, clip_range_vf)
        # Value loss using the TD(gae_lambda) target
        return tf.keras.losses.MSE(return_batch, values_pred)

    def train(self, gradient_steps, batch_size=40):
        # Update optimizer learning rate
        # self._update_learning_rate(self.policy.optimizer)

        # Compute current clip range
        clip_range = self.clip_range
        # if self.clip_range_vf is not None:
        clip_range_vf = self.clip_range_vf
        # else:
        #clip_range_vf = None

        for gradient_step in range(gradient_steps):
            approx_kl_divs = []
            # Sample replay buffer
            for replay_data in self.rollout_buffer.get(batch_size):
                # Unpack
                obs, action, old_values, old_log_prob, advantage, return_batch = replay_data

                # if isinstance(self.action_space, spaces.Discrete):
                #     # Convert discrete action for float to long
                #     action = action.astype(np.int64).flatten()

                with tf.GradientTape() as tape:
                    tape.watch(self.policy.trainable_variables)
                    values, log_prob, entropy = self.policy.evaluate_actions(
                        obs, action)
                    # Flatten
                    values = tf.reshape(values, [-1])

                    policy_loss = self.policy_loss(
                        advantage, log_prob, old_log_prob, clip_range)
                    value_loss = self.value_loss(
                        values, old_values, return_batch, clip_range_vf)

                    # Entropy loss favor exploration
                    entropy_loss = -tf.reduce_mean(entropy)

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                gradients = tape.gradient(
                    loss, self.policy.trainable_variables)
                # Clip grad norm
                # gradients = tf.clip_by_norm(gradients, self.max_grad_norm)
                self.policy.optimizer.apply_gradients(
                    zip(gradients, self.policy.trainable_variables))
                approx_kl_divs.append(tf.reduce_mean(
                    old_log_prob - log_prob).numpy())

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print("Early stopping at step {} due to reaching max kl: {:.2f}".format(gradient_step,
                                                                                        np.mean(approx_kl_divs)))
                break

        # explained_var = explained_variance(self.rollout_buffer.returns.flatten(),
        #                                    self.rollout_buffer.values.flatten())

    # def learn(self, ):
    #     observations = self.env.reset()
    #
    #     [obz_0, obz_1] = [observations.....]
    #
    #     while self.num_timesteps < total_timesteps:
    #
    #
    #         obs = self.collect_rollouts(self.env, self.rollout_buffer, n_rollout_steps=self.n_steps,
    #                                     obs=obz_0)
    #         self.num_timesteps += self.n_steps * self.n_envs
    #         timesteps_since_eval += self.n_steps * self.n_envs
    #         #self._update_current_progress(self.num_timesteps, total_timesteps)
    #
    #         # Display training infos
    #
    #
    #         self.train(self.n_epochs, batch_size=self.batch_size)
    #
    #
    #
    #     return self

    def learn(self, max_epochs):

        epoch = 0

        while epoch <= max_epochs:
            self.env.seed(self.seed)
            observations = self.env.reset()
            print(f"Rollouts: {epoch}")

            flag = self.collect_rollouts(self.env, self.rollout_buffer, n_rollout_steps=self.n_steps,
                                         obs=observations)
            if flag == 1:
                print(f"Epochs:{epoch} End...\n")
                break

            self.train(self.n_epochs, batch_size=self.batch_size)
            epoch += 1
