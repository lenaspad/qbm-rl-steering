# Loosely based on:
# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MSE

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from qbm_rl_steering.core.utils import (generate_classical_critic,
                                        generate_classical_actor)
from qbm_rl_steering.core.utils import ReplayBuffer
from qbm_rl_steering.core.qbm import QFunction


# TODO: implement common base class for classical and quantum ddpg
# TODO: needs further cleaning: get rid of n_steps_estimate and implement the
#  learning rate schedules properly.


class ClassicalDDPG:
    def __init__(self, state_space, action_space, gamma, tau_critic, tau_actor,
                 learning_rate_critic, learning_rate_actor, grad_clip_actor,
                 grad_clip_critic):
        """ Implements the classical DDPG agent where both actor and critic
        are represented by classical neural networks.
        :param state_space: openAI gym env state space
        :param action_space: openAI gym env action space
        :param gamma: reward discount factor
        :param tau_critic: soft update factor for critic target network
        :param tau_actor: soft update factor for actor target network
        :param learning_rate_schedule_critic: learning rate schedule for critic.
        :param learning_rate_schedule_actor: learning rate schedule for actor.
        """
        self.step_count = 0

        self.n_dims_state_space = len(state_space.high)
        self.n_dims_action_space = len(action_space.high)

        # Some main hyperparameters
        self.gamma = gamma
        self.tau_critic = tau_critic
        self.tau_actor = tau_actor

        self.grad_clip_actor = grad_clip_actor
        self.grad_clip_critic = grad_clip_critic

        # Main and target actor network initialization
        # ACTOR
        # actor_hidden_layers = [48, 32]
        actor_hidden_layers = [400, 300]
        self.main_actor_net = generate_classical_actor(
            self.n_dims_state_space, self.n_dims_action_space, actor_hidden_layers)
        self.target_actor_net = generate_classical_actor(
            self.n_dims_state_space, self.n_dims_action_space, actor_hidden_layers)

        # CRITIC
        # critic_hidden_layers = [48, 32, 1]
        critic_hidden_layers = [400, 300, 1]
        self.main_critic_net_1 = generate_classical_critic(
            self.n_dims_state_space, self.n_dims_action_space, critic_hidden_layers)
        # self.main_critic_net_2 = generate_classical_critic(
        #     self.n_dims_state_space, self.n_dims_action_space)
        self.target_critic_net_1 = generate_classical_critic(
            self.n_dims_state_space, self.n_dims_action_space, critic_hidden_layers)
        # self.target_critic_net_2 = generate_classical_critic(
        #     self.n_dims_state_space, self.n_dims_action_space)

        # Copy weights from main to target nets
        self.target_actor_net.set_weights(self.main_actor_net.get_weights())
        self.target_critic_net_1.set_weights(self.main_critic_net_1.get_weights())
        # self.target_critic_net_2.set_weights(
        #     self.main_critic_net_2.get_weights())

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate_actor, amsgrad=False, epsilon=1e-8, beta_1=0.9, beta_2=0.999)
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate_critic, amsgrad=False, epsilon=1e-8, beta_1=0.9, beta_2=0.999)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            size=int(1e6), obs_dim=self.n_dims_state_space,
            act_dim=self.n_dims_action_space)
        # self.replay_buffer = ReplayBuffer(buffer_size=int(1e6))

        # For logging
        # TODO: add critic gradient statistics as well?
        self.critic_grads_log = {'mean': [], 'min': [], 'max': []}
        self.actor_grads_log = {'mean': [], 'min': [], 'max': []}
        self.losses_log = {'Q': [], 'Mu': []}
        self.q_log = {'before': [], 'after': []}

    def get_proposed_action(self, state):
        """ Use main actor network to obtain proposed action for en input
        state. """
        return self.main_actor_net.predict(state.reshape(1, -1))[0]
        # action += noise_scale * np.random.randn(self.n_dims_action_space)
        # return np.clip(action, -1., 1.)

    def update(self, batch_size, *args):
        """ Calculate and apply the updates of the critic and actor
        networks based on batch of samples from experience replay buffer. """
        s, a, r, s2, d = self.replay_buffer.sample(batch_size)
        s = np.asarray(s, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        r = np.asarray(r, dtype=np.float32)
        s2 = np.asarray(s2, dtype=np.float32)
        d = np.asarray(d, dtype=np.float32)

        grads_critic_1 = self._get_gradients_critic(s, a, r, s2, d)
        self.critic_optimizer.apply_gradients(
            zip(grads_critic_1, self.main_critic_net_1.trainable_variables))
        # self.critic_optimizer.apply_gradients(
        #     zip(grads_critic_2, self.main_critic_net_2.trainable_variables))

        # TD3 feature: delay
        if self.step_count % 1 == 0:
            grads_actor = self._get_gradients_actor(s, batch_size)
            self.actor_optimizer.apply_gradients(
                zip(grads_actor, self.main_actor_net.trainable_variables))

            # This is for debugging only, i.e. not required for algorithm
            # to work: evaluate Q value for actor after update: providing same
            # state, should now evaluate to higher Q.
            a_mu_after = self.main_actor_net(s)
            q_mu_after = self.main_critic_net_1([s, a_mu_after])
            self.q_log['after'].append(np.mean(q_mu_after))

            # Apply Polyak updates
            self._update_target_networks()

        self.step_count += 1

    def _get_gradients_critic(self, state, action, reward, next_state, d):
        """ Update the main critic network based on given batch of input
        states. """
        with tf.GradientTape() as tape:
            next_action = self.target_actor_net(next_state)
            # next_action = np.clip(next_action, -1, 1)

            # TD3: add noise to next_action
            # Select action according to policy and add clipped noise
            # policy_noise = 0.2
            # noise_clip = 0.4
            # noise = policy_noise * np.random.randn(
            #     next_action.shape[0] * next_action.shape[1]).reshape(
            #     next_action.shape[0], next_action.shape[1])
            # noise = np.clip(noise, a_min=-noise_clip, a_max=noise_clip)
            # next_action = np.clip(next_action + noise, -1., 1.)

            q1 = self.target_critic_net_1([next_state, next_action])
            # q2 = self.target_critic_net_2([next_state, next_action])
            # q_target = np.min(np.stack((q1, q2)), axis=0)
            # q_target = (reward + self.gamma * (1. - d) * q_target)
            q_target = (reward + self.gamma * (1. - d) * q1)

            q_vals_1 = self.main_critic_net_1([state, action])
            # q_vals_2 = self.main_critic_net_2([state, action])

            # q_loss = tf.reduce_mean((q_vals - q_target) ** 2)
            # q_loss = MSE(q_target, q_vals)
            # q_loss = tf.reduce_mean(tf.math.abs(q_vals - q_target))
            q_loss_1 = tf.math.reduce_mean((q_target - q_vals_1) ** 2)
            # q_loss_2 = tf.reduce_mean((q_vals_2 - q_target)**2)

        grads_q_1 = tape.gradient(q_loss_1, self.main_critic_net_1.trainable_variables)
        # grads_q_1 = [tf.clip_by_value(
        #     grad, -self.grad_clip_critic, self.grad_clip_critic) for grad in
        #     grads_q_1]

        # grads_q_2 = None
        # grads_q_2 = tape.gradient(
        #     q_loss_2, self.main_critic_net_2.trainable_variables)
        # grads_q_2 = [tf.clip_by_value(
        #     grad, -self.grad_clip_critic, self.grad_clip_critic) for grad in
        #     grads_q_2]

        # Have created tape as persistent, need to clean up manually
        self.losses_log['Q'].append(q_loss_1)

        # Just for logging
        mean_ = 0.
        min_ = 1E39
        max_ = -1E39
        for grd in grads_q_1:
            mean_ += np.mean(grd)
            min_ = min(np.min(grd), min_)
            max_ = max(np.max(grd), max_)
        mean_ /= len(grads_q_1)

        self.critic_grads_log['min'].append(min_)
        self.critic_grads_log['max'].append(max_)
        self.critic_grads_log['mean'].append(mean_)

        return grads_q_1  # , grads_q_2

    def _get_gradients_actor(self, states, batch_size):
        """ Update the main actor network based on given batch of input
        states. """
        with tf.GradientTape() as tape2:
            A_mu = self.main_actor_net(states)
            Q_mu = self.main_critic_net_1([states, A_mu])
            self.q_log['before'].append(np.mean(Q_mu))
            mu_loss = -tf.math.reduce_mean(Q_mu)
        grads_mu = tape2.gradient(mu_loss,
                                  self.main_actor_net.trainable_variables)
        # grads_mu = [tf.clip_by_value(
        #     grad, -self.grad_clip_actor, self.grad_clip_actor) for grad in
        #     grads_mu]
        self.losses_log['Mu'].append(mu_loss)

        # The same thing as above implemented manually to be as close to
        # QuantumDDPG implementation as possible
        # TODO: find out where the tensorflow warning comes from and try to
        #  speed up the code ...
        # with tf.GradientTape() as tape2:
        #     a_mu = self.main_actor_net(states)
        # jacobi_mu_wrt_mu_theta = tape2.jacobian(
        #     a_mu, self.main_actor_net.trainable_variables
        # )
        #
        # with tf.GradientTape() as tape3:
        #     tape3.watch(a_mu)
        #     q_mu = self.main_critic_net([states, a_mu])
        #     self.q_log['before'].append(np.mean(q_mu))
        #
        # grad_q_wrt_a = tape3.gradient(q_mu, a_mu)
        #
        # grads_mu = []
        # for i in range(len(jacobi_mu_wrt_mu_theta)):
        #     on_batch = tf.tensordot(
        #         jacobi_mu_wrt_mu_theta[i], grad_q_wrt_a,
        #         axes=((0, 1), (0, 1)))
        #     on_batch /= batch_size
        #     grads_mu.append(-on_batch)

        # Just for logging
        mean_ = 0.
        min_ = 1E39
        max_ = -1E39
        for grd in grads_mu:
            mean_ += np.mean(grd)
            min_ = min(np.min(grd), min_)
            max_ = max(np.max(grd), max_)
        mean_ /= len(grads_mu)

        self.actor_grads_log['min'].append(min_)
        self.actor_grads_log['max'].append(max_)
        self.actor_grads_log['mean'].append(mean_)

        return grads_mu

    def _update_target_networks(self):
        """ Apply Polyak update to both target networks. """
        # CRITICS
        target_weights = np.array(self.target_critic_net_1.get_weights())
        main_weights = np.array(self.main_critic_net_1.get_weights())
        new_target_weights = (self.tau_critic * main_weights +
                              (1 - self.tau_critic) * target_weights)
        self.target_critic_net_1.set_weights(new_target_weights)

        # target_weights = np.array(self.target_critic_net_2.get_weights())
        # main_weights = np.array(self.main_critic_net_2.get_weights())
        # new_target_weights = (self.tau_critic * main_weights +
        #                       (1 - self.tau_critic) * target_weights)
        # self.target_critic_net_2.set_weights(new_target_weights)

        # ACTOR
        target_weights = np.array(self.target_actor_net.get_weights())
        main_weights = np.array(self.main_actor_net.get_weights())
        new_target_weights = (self.tau_actor * main_weights +
                              (1 - self.tau_actor) * target_weights)
        self.target_actor_net.set_weights(new_target_weights)


class QuantumDDPG:
    def __init__(self, state_space, action_space, gamma, tau_critic,
                 tau_actor, learning_rate_schedule_critic,
                 learning_rate_schedule_actor, grad_clip_actor=np.inf,
                 grad_clip_critic=np.inf, n_replicas=2, big_gamma=(25., 0.5),
                 beta=0.5, n_annealing_steps=100, n_meas_for_average=1,
                 n_rows_qbm=3, n_columns_qbm=3):
        """ Implements quantum DDPG where actor and critic networks are
        represented by quantum Boltzmann machines and classical neural
        networks, respectively.
        :param state_space: openAI gym env state space
        :param action_space: openAI gym env action space
        :param gamma: reward discount factor
        :param tau_critic: soft update factor for critic target network
        :param tau_actor: soft update factor for actor target network
        :param learning_rate_schedule_critic: learning rate schedule for critic.
        :param learning_rate_schedule_actor: learning rate schedule for actor.
        :param grad_clip_actor: limits to which actor gradients are clipped
        :param grad_clip_critic: limits to which critic gradients are clipped
        :param n_steps_estimate: estimated total number of training steps,
        needed for learning rate schedules (would like to get rid of this).
        """
        self.n_dims_state_space = len(state_space.high)
        self.n_dims_action_space = len(action_space.high)

        # Main hyperparameters
        self.gamma = gamma
        self.tau_critic = tau_critic
        self.tau_actor = tau_actor

        # Learning rate schedules
        self.lr_schedule_critic = learning_rate_schedule_critic
        self.lr_schedule_actor = learning_rate_schedule_actor

        kwargs_q_func = dict(
            sampler_type='QPU',
            state_space=state_space,
            action_space=action_space,
            small_gamma=self.gamma,
            n_replicas=n_replicas,
            big_gamma=big_gamma, beta=beta,   # (20., 0.), beta = 2.  ... (best? (500., 2.), beta = 0.1
            n_annealing_steps=n_annealing_steps,  # 200  (best? 10'000)
            n_meas_for_average=n_meas_for_average,
            kwargs_qpu={},
            n_rows=n_rows_qbm,
            n_columns=n_columns_qbm)

        # Gradient clipping limits
        self.grad_clip_actor = grad_clip_actor
        self.grad_clip_critic = grad_clip_critic

        # Main and target actor network initialization
        # ACTOR
        # actor_hidden_layers = [200, 150]
        actor_hidden_layers = [400, 300]
        self.main_actor_net = generate_classical_actor(
            self.n_dims_state_space, self.n_dims_action_space,
            actor_hidden_layers)
        self.target_actor_net = generate_classical_actor(
            self.n_dims_state_space, self.n_dims_action_space,
            actor_hidden_layers)

        # CRITIC
        self.main_critic_net = QFunction(**kwargs_q_func, qfunc_it=0)
        self.target_critic_net = QFunction(**kwargs_q_func, qfunc_it=1)

        # Copy weights of main networks onto targets
        self.target_actor_net.set_weights(self.main_actor_net.get_weights())
        for k in self.main_critic_net.w_hh.keys():
            self.target_critic_net.w_hh[k] = self.main_critic_net.w_hh[k]
        for k in self.target_critic_net.w_vh.keys():
            self.target_critic_net.w_vh[k] = self.main_critic_net.w_vh[k]

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate_schedule_actor, amsgrad=False, epsilon=1e-8, beta_1=0.9, beta_2=0.999)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            size=int(1e6), obs_dim=self.n_dims_state_space,
            act_dim=self.n_dims_action_space)
        # self.replay_buffer = ReplayBuffer(buffer_size=int(1e6))

        # For logging
        # TODO: add critic gradient statistics as well?
        self.actor_grads_log = {'mean': [], 'min': [], 'max': []}
        self.losses_log = {'Q': [], 'Mu': []}
        self.q_log = {'before': [], 'after': []}

    def get_proposed_action(self, state):
        """ Use main actor network to obtain proposed action for en input
        state. """
        return self.main_actor_net.predict(state.reshape(1, -1))[0]
        # action += noise_scale * np.random.randn(self.n_dims_action_space)
        # return np.clip(action, -1., 1.)

    def update(self, batch_size, step_count, delay_actor_update=1):
        """ Calculate and apply the updates of the critic and actor
        networks based on batch of samples from experience replay buffer. """
        s, a, r, s2, d = self.replay_buffer.sample(batch_size)
        s = np.asarray(s, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        r = np.asarray(r, dtype=np.float32)
        s2 = np.asarray(s2, dtype=np.float32)
        d = np.asarray(d, dtype=np.float32)

        # Invert order since _update_critic will directly apply gradient update
        # while _get_gradients_actor does not.
        if step_count % delay_actor_update == 0:
            grads_actor = self._get_gradients_actor(s, batch_size)
        self._update_critic(s, a, r, s2, d, step_count)

        if step_count % delay_actor_update == 0:
            self.actor_optimizer.apply_gradients(
                zip(grads_actor, self.main_actor_net.trainable_variables))

        # Evaluate Q value for new actor (providing same state, should now
        # give higher Q)
        # a_mu_after = self.main_actor_net(s)
        # q_after, _, _ = self.main_critic_net.calculate_q_value_on_batch(
        #     s, a_mu_after)
        # self.q_log['after'].append(np.mean(q_after))

        # Apply Polyak updates
        self._update_target_critic()
        if step_count % delay_actor_update == 0:
            self._update_target_actor()

    def _update_critic(self, state, action, reward, next_state,
                       done, step_count):
        """ Update the main critic network based on given batch of input
        states. """
        next_action = self.target_actor_net(next_state)

        # Loop over batch of states, actions, rewards, next_states
        print('Looping over batch of samples, getting current and future Q values in _update_critic ...')
        for jj in np.arange(len(state)):
            if jj % 4 == 0:
                print(f'Sample in batch: {jj + 1}/{len(state)}')

            # Recalculate q_value of (sample.state, sample.action) pair
            q_value, spin_configs, visible_nodes = (
                self.main_critic_net.calculate_q_value(state[jj], action[jj]))

            # Now calculate the target_q_value of action proposed by the actor
            target_q_value, spin_configurations, visible_nodes = (
                self.target_critic_net.calculate_q_value(
                    state=next_state[jj], action=next_action[jj]))

            # Update weights and target Q-function if needed
            # Note that clipping is also done inside QBM update_weights
            # method ...
            self.main_critic_net.update_weights(
                spin_configs, visible_nodes, q_value,
                (1. - done[jj]) * target_q_value,
                reward[jj],
                learning_rate=self.lr_schedule_critic(step_count).numpy(),
                grad_clip=self.grad_clip_critic)

    def _get_gradients_actor(self, state, batch_size):
        """ Update the main actor network based on given batch of input
        states. """
        with tf.GradientTape() as tape2:
            a_mu = self.main_actor_net(state)

        # print('Calling main_critic to calculate q values on batch in _get_gradients_actor ...')
        # q, _, _ = self.main_critic_net.calculate_q_value_on_batch(state, a_mu)
        # self.q_log['before'].append(np.mean(q))
        # self.losses_log['Q'].append(np.mean(q))

        # Apply chain-rule manually here:
        jacobi_mu_wrt_mu_theta = tape2.jacobian(
            a_mu, self.main_actor_net.trainable_variables)
        # grad_q_wrt_a = self.get_action_derivative(state, a_mu, batch_size)
        # print('numerical: ', grad_q_wrt_a)
        grad_q_wrt_a = self.get_analytical_action_derivative(state, a_mu)
        # print('analytical: ', grad_q_wrt_a)

        grads_mu = []
        for i in range(len(jacobi_mu_wrt_mu_theta)):
            on_batch = tf.tensordot(
                jacobi_mu_wrt_mu_theta[i], grad_q_wrt_a,
                axes=((0, 1), (0, 1)))
            on_batch /= batch_size

            on_batch = np.clip(on_batch,
                               a_min=-self.grad_clip_actor,
                               a_max=self.grad_clip_actor)
            grads_mu.append(-on_batch)

        # For logging
        mean_ = 0.
        min_ = 1E39
        max_ = -1E39
        for grd in grads_mu:
            mean_ += np.mean(grd)
            min_ = min(np.min(grd), min_)
            max_ = max(np.max(grd), max_)
        mean_ /= len(grads_mu)

        self.actor_grads_log['min'].append(min_)
        self.actor_grads_log['max'].append(max_)
        self.actor_grads_log['mean'].append(mean_)

        return grads_mu

    def _update_target_critic(self):
        """ Apply Polyak update to both target networks. """
        # CRITIC
        for k in self.main_critic_net.w_hh.keys():
            self.target_critic_net.w_hh[k] = (
                    self.tau_critic * self.main_critic_net.w_hh[k] +
                    (1. - self.tau_critic) * self.target_critic_net.w_hh[k])
        for k in self.main_critic_net.w_vh.keys():
            self.target_critic_net.w_vh[k] = (
                    self.tau_critic * self.main_critic_net.w_vh[k] +
                    (1. - self.tau_critic) * self.target_critic_net.w_vh[k])

    def _update_target_actor(self):
        # ACTOR
        target_weights = np.array(self.target_actor_net.get_weights())
        main_weights = np.array(self.main_actor_net.get_weights())
        new_target_weights = (self.tau_actor * main_weights +
                              (1 - self.tau_actor) * target_weights)
        self.target_actor_net.set_weights(new_target_weights)

    def get_action_derivative(
            self, states: list, actions: list, batch_size: int,
            epsilon: float = 0.01):  # eps. default 0.3. 0.01 also worked well :) !
        """ Calculate numerical derivative of QBM Q values with respect to
        action.
        :param states: batch of states from replay buffer.
        :param actions: corresponding actions.
        :param batch_size: number of samples drawn from replay buffer.
        :param epsilon: step size to take numerical derivative using shift
        rule.
        :return np.ndarray of gradients of shape (batch_size,
        n_dims_action_space). """
        # Need to take derivative for each action separately
        # e.g. if we have batch size of 5, and 10 actions, we expect an
        # output for dQ / da of shape (5, 10).
        # grads_mean = np.zeros((batch_size, self.action_dim))

        print('Calling main_critic 2x n_action_dims to calculate q values on batch in get_action_derivative ...')

        n_avg = 1
        grads = np.zeros((n_avg, batch_size, self.n_dims_action_space))
        for j in range(n_avg):
            for i in range(self.n_dims_action_space):
                actions_tmp1 = np.array(actions).copy()
                actions_tmp1[:, i] += epsilon
                qeps_plus, _, _ = (
                    self.main_critic_net.calculate_q_value_on_batch(
                        states, actions_tmp1)
                )

                actions_tmp2 = np.array(actions).copy()
                actions_tmp2[:, i] -= epsilon
                qeps_minus, _, _ = (
                    self.main_critic_net.calculate_q_value_on_batch(
                        states, actions_tmp2)
                )

                grads[j, :, i] = np.atleast_1d(
                    np.float_((qeps_plus - qeps_minus) / (2 * epsilon)))

        grads = np.asarray(grads, dtype=np.float32)
        grads = np.mean(grads, axis=0)
        return grads

    def get_analytical_action_derivative(self, states, actions):
        print('Calling main_critic to calculate q values on batch in get_analytical_action_derivative ...')
        _, _, _, _, grads = self.main_critic_net.calculate_q_value_on_batch(
            states, np.array(actions), calc_derivative=True)
        return np.asarray(-grads, dtype=np.float32)
