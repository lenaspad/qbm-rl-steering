import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm
import pickle

from qbm_rl_steering.core.ddpg_agents import QuantumDDPG


def trainer(env, agent, n_steps, max_steps_per_episode, batch_size,
            action_noise_schedule, epsilon_greedy_schedule, n_anneals_schedule,
            n_exploration_steps=30, n_episodes_early_stopping=20, out_path='./'):
    """ Convenience function to run training with DDPG.
    :param env: openAI gym environment instance
    :param agent: ddpg instance (ClassicalDDPG or QuantumDDPG)
    :param n_steps: max. number of steps that training will run for
    :param max_steps_per_episode: max. number of steps allowed per episode
    :param batch_size: number of samples drawn from experience replay buffer
    at every step.
    :param action_noise_schedule: action noise decay over time.
    :param epsilon_greedy_schedule: epsilon-greedy schedule. Decides what
    fraction of actions will be purely random.
    :param n_anneals_schedule: number of anneals as training progresses.
    :param n_exploration_steps: number of initial random steps in env.
    :param n_episodes_early_stopping: number of consecutive episodes with
    certain number of steps (< 3) to count towards early stopping.
    :return tuple of init, final rewards, number of steps, and random steps
    (all per episode).
    """
    episode_log = {
        'initial_rewards': [], 'final_rewards': [], 'n_total_steps': [],
        'n_random_steps': [], 'max_steps_above_reward_threshold': []
    }

    n_total_steps_training = 0 
    early_stopping_counter = 0
    k_moving_avg = 30

    max_act = -np.inf
    min_act = np.inf
    max_state = -np.inf
    min_state = np.inf

    episode = 0
    while n_total_steps_training < n_steps:

        if early_stopping_counter >= n_episodes_early_stopping:
            break

        n_random_steps_episode = 0
        n_steps_episode = 0

        # state = env.reset(init_outside_threshold=False)
        state = env.reset(init_outside_threshold=True)
        try:
            episode_log['initial_rewards'].append(env.calculate_reward(
                env.calculate_state(env.kick_angles)))
        except AttributeError:
            episode_log['initial_rewards'].append(env._get_reward(state))

        # Apply n_anneals_schedule
        n_anneals = int(n_anneals_schedule(n_total_steps_training).numpy())
        if isinstance(agent, QuantumDDPG):
            agent.main_critic_net.n_meas_for_average = n_anneals
            agent.target_critic_net.n_meas_for_average = n_anneals
        else:
            agent.main_critic_net_1.n_meas_for_average = n_anneals
            agent.target_critic_net_1.n_meas_for_average = n_anneals
            # agent.main_critic_net_2.n_meas_for_average = n_anneals
            # agent.target_critic_net_2.n_meas_for_average = n_anneals

        # Episode loop
        epsilon = epsilon_greedy_schedule(n_total_steps_training).numpy()
        
        # Action noise decay
        action_noise = action_noise_schedule(n_total_steps_training).numpy()

        for _ in range(max_steps_per_episode):
            eps_sample = np.random.uniform(0, 1, 1)
            if ((n_total_steps_training < n_exploration_steps) or
                    (eps_sample <= epsilon)):
                action = env.action_space.sample()
                n_random_steps_episode += 1
            else:
                action = agent.get_proposed_action(state)

            max_act = max(np.max(action), max_act)
            min_act = min(np.min(action), min_act)

            # NEW: ADD ACTION NOISE IN ANY CASE, DURING RANDOM EXPLORATION AND WHEN SAMPLING
            # FOLLOWING POLICY.
            action += action_noise * np.random.randn(agent.n_dims_action_space)
            action = np.clip(action, -1., 1.)

            next_state, reward, done, _ = env.step(action)

            max_state = max(np.max(next_state), max_state)
            min_state = min(np.min(next_state), min_state)

            n_steps_episode += 1
            n_total_steps_training += 1

            # Fill replay buffer
            terminal = done
            if n_steps_episode == max_steps_per_episode:
                done = True
                terminal = False
            agent.replay_buffer.push(state, action, reward, next_state,
                                     terminal)

            # EPISODE DONE
            if done:
                # Append log
                episode_log['final_rewards'].append(reward)
                episode_log['n_total_steps'].append(n_steps_episode)
                episode_log['n_random_steps'].append(n_random_steps_episode)
                # episode_log['max_steps_above_reward_threshold'].append(
                #     env.max_steps_above_reward_threshold)

                # Early stopping counter
                # if (((n_steps_episode - n_random_steps_episode) <=
                #      (env.required_steps_above_reward_threshold + 1)) and
                #         reward > env.reward_threshold and
                #         n_steps_episode != n_random_steps_episode):
                try:
                    rew_thresh = env.reward_threshold
                except AttributeError:
                    rew_thresh = env.threshold

                steps_above_thresh = 0
                try:
                    steps_above_thresh = env.required_steps_above_reward_threshold
                except AttributeError:
                    pass

                if (((n_steps_episode - n_random_steps_episode) <=
                     (steps_above_thresh + 2)) and
                        reward > rew_thresh):  # and
                    # n_steps_episode != n_random_steps_episode):
                    early_stopping_counter += 1
                else:
                    early_stopping_counter = 0

                # Print episode data
                moving_average_final_rewards = np.mean(
                    np.array(episode_log['final_rewards'][-k_moving_avg:]))
                moving_average_n_steps = np.mean(
                    np.array(episode_log['n_total_steps'][-k_moving_avg:]))

                print(f"\nEPISODE: {episode}")
                print(
                    f"INITIAL REWARD: "
                    f"{round(episode_log['initial_rewards'][-1], 3)}\n"
                    f"FINAL REWARD: "
                    f"{round(episode_log['final_rewards'][-1], 3)}\n"
                    f"#STEPS: {n_steps_episode} "
                    f"({n_random_steps_episode} RANDOM)\n"
                    # f"MAX STEPS ABOVE REW. THRESH.: "
                    # f"{env.max_steps_above_reward_threshold}\n"
                    # f"ABORT REASON: "
                    # f"{env.interaction_logger.log_episode[-1][-1]}\n"
                    f"EARLY STOPPING COUNT: "
                    f"{early_stopping_counter}/{n_episodes_early_stopping}\n\n"
                    f"MOVING AVG FINAL REWARD: "
                    f"{np.round(moving_average_final_rewards, 3)}\n"
                    f"MOVING AVG #STEPS: "
                    f"{np.round(moving_average_n_steps, 1)}"
                )
                episode += 1

                if n_total_steps_training > n_exploration_steps:
                    # TRAINING ON RANDOM BATCHES FROM REPLAY BUFFER
                    # (for n_steps_episode)
                    tqdm_desc = f'Learning progress -- Episode {episode}'
                    for stp in tqdm.trange(n_steps_episode, desc=tqdm_desc):
                        agent.update(batch_size, n_total_steps_training)

                        # Save agent, buffer, and training log after every training step
                        print(f'SAVING EVERYTHING: Ep {episode}, stp {stp}')
                        weights = {'main_critic': {'w_vh': agent.main_critic_net.w_vh, 'w_hh': agent.main_critic_net.w_hh},
                                   'target_critic': {'w_vh': agent.target_critic_net.w_vh, 'w_hh': agent.target_critic_net.w_hh}}
                        with open(out_path + f'/critic_weights_ep{episode}_stp{stp}.pkl', 'wb') as fid:
                            pickle.dump(weights, fid)

                        weights = {'main_actor': agent.main_actor_net.get_weights(),
                                   'target_actor': agent.target_actor_net.get_weights()}
                        with open(out_path + f'/actor_weights_ep{episode}_stp{stp}.pkl', 'wb') as fid:
                            pickle.dump(weights, fid)

                        with open(out_path + f'/episode_log.pkl', 'wb') as fid:
                            pickle.dump(episode_log, fid)

                        with open(out_path + f'/replay_buffer.pkl', 'wb') as fid:
                            pickle.dump(agent.replay_buffer, fid)

                break

            state = next_state

    print('n_total_steps_training', n_total_steps_training)
    return episode_log


def evaluator(env, agent, n_episodes, reward_scan=True):
    """ Run trained agent for a number of episodes.
    :param env: openAI gym based environment
    :param agent: trained agent
    :param n_episodes: number of episodes for evaluation.
    :param reward_scan: if False, init episodes randomly. If True, run scan
    in specific range of rewards.
    :return ndarray with all rewards. """

    all_rewards = []
    if reward_scan:
        kick_max = np.array([env.kick_angle_max] * env.n_dims)
        kick_min = np.array([env.kick_angle_min] * env.n_dims)
        min_reward = 0.9 * min(
            env.calculate_reward(env.calculate_state(kick_max)),
            env.calculate_reward(env.calculate_state(kick_min)))

    episode = 0
    while episode < n_episodes:
        if reward_scan:
            target_init_rewards = np.linspace(
                min_reward, env.reward_threshold, n_episodes)
            state = env.reset(
                init_specific_reward_state=target_init_rewards[episode])
        else:
            state = env.reset(init_outside_threshold=True)

        rewards = []
        try:
            rewards.append(env.calculate_reward(
                env.calculate_state(env.kick_angles)))
        except AttributeError:
            rewards.append(env._get_reward(state))

        # rewards = [env.calculate_reward(
        #     env.calculate_state(env.kick_angles))]

        n_steps_eps = 0
        while True:
            try:
                a = agent.get_proposed_action(state)
            except:
                a, _ = agent.predict(state, deterministic=True)
            state, reward, done, _ = env.step(a)
            rewards.append(reward)
            if done:
                episode += 1
                all_rewards.append(rewards)
                break
            n_steps_eps += 1

    return np.array(all_rewards)


def plot_training_log(env, agent, data, save_path=None, apply_scaling=False):
    """ Plot the log data from the training. """
    # a) Training log
    fig1, axs = plt.subplots(2, 1, sharex=True)

    n_training_episodes = len(data['final_rewards'])

    if apply_scaling:
        # Undo all scalings that have been applied to the reward and multiply by
        # 1'000 to get from [m] -> [mm]
        scaling = 1. / (env.state_scale * env.reward_scale) * 1000
    else:
        scaling = 1.

    axs[0].plot(data['n_total_steps'], label='Total steps')
    axs[0].plot(data['n_random_steps'], '--', c='k',
                label='Random steps')
    axs[1].plot(np.array(data['initial_rewards'])[-n_training_episodes:] * scaling, c='r',
                label='Initial')
    axs[1].plot(np.array(data['final_rewards']) * scaling, c='g', label='Final')
    try:
        axs[1].axhline(env.reward_threshold * scaling, color='k', ls='--')
    except AttributeError:
        axs[1].axhline(env.threshold * scaling, color='k', ls='--')

    axs[0].set_ylabel('Number of steps')
    axs[1].set_ylabel('Reward (mm)')
    axs[1].set_xlabel('Episodes')
    axs[0].legend(loc='upper right')
    axs[1].legend(loc='lower left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + '/train_log_steps_rewards.png', dpi=150)
        plt.close()
    else:
        plt.show()

    # b) AgentL q before vs after
    fig2 = plt.figure()
    plt.plot(np.array(agent.q_log['before']), c='r', label='q before')
    plt.plot(np.array(agent.q_log['after']), c='g', label='q after')
    plt.legend()
    plt.ylabel('Q value')
    plt.xlabel('Training iterations')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + '/train_log_qbefore_after.png', dpi=150)
        plt.close()
    else:
        plt.show()

    # c) Agent: q_after minus q_before
    fig3 = plt.figure()
    plt.plot(np.array(agent.q_log['after']) - np.array(agent.q_log['before']))
    plt.ylabel(r'$Q_{{f}} - Q_{{i}}$')
    plt.xlabel('Training iterations')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + '/train_log_qdelta.png', dpi=150)
        plt.close()
    else:
        plt.show()

    # d) Agent: gradients evolution
    fig4 = plt.figure()
    plt.plot(agent.actor_grads_log['mean'], label='mean')
    plt.plot(agent.actor_grads_log['min'], label='min')
    plt.plot(agent.actor_grads_log['max'], label='max')
    plt.ylabel('Actor grads')
    plt.xlabel('Training iterations')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + '/train_log_actor_grads.png', dpi=150)
        plt.close()
    else:
        plt.show()

    if not isinstance(agent, QuantumDDPG):
        fig41 = plt.figure()
        plt.plot(agent.critic_grads_log['mean'], label='mean')
        plt.plot(agent.critic_grads_log['min'], label='min')
        plt.plot(agent.critic_grads_log['max'], label='max')
        plt.ylabel('Critic grads')
        plt.xlabel('Training iterations')
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path + '/train_log_critic_grads.png', dpi=150)
            plt.close()
        else:
            plt.show()

    # e) Training evolution of final reward and #steps
    win = 10
    rew_padded = np.pad(np.array(data['final_rewards']) * scaling,
                        (win // 2, win - 1 - win // 2), mode='edge')
    ds_rew = pd.Series(rew_padded)

    steps_padded = np.pad(np.array(data['n_total_steps']),
                          (win // 2, win - 1 - win // 2), mode='edge')
    ds_steps = pd.Series(steps_padded)

    fig5, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    # Final reward
    reward_mean = ds_rew.rolling(win).mean().dropna().reset_index(drop=True)
    reward_std = (ds_rew.rolling(win).std().dropna().reset_index(drop=True) /
                  np.sqrt(win))
    axs[0].plot(np.array(data['final_rewards']) * scaling, lw=1.5, c='tab:green', alpha=0.7)
    axs[0].fill_between(np.arange(len(reward_mean)),
                        reward_mean - reward_std, reward_mean + reward_std,
                        alpha=0.4, color='tab:green')
    axs[0].plot(reward_mean, lw=2, c='tab:green')

    # Steps
    steps_mean = ds_steps.rolling(win).mean().dropna().reset_index(drop=True)
    steps_std = (ds_steps.rolling(win).std().dropna().reset_index(drop=True) /
                 np.sqrt(win))
    axs[1].plot(data['n_total_steps'], lw=1.5, c='tab:blue', alpha=0.7)
    axs[1].fill_between(np.arange(len(steps_mean)),
                        steps_mean - steps_std, steps_mean + steps_std,
                        alpha=0.4, color='tab:blue')
    axs[1].plot(steps_mean, lw=2, c='tab:blue')

    axs[0].set_ylabel('Final reward (mm)')
    axs[1].set_ylabel('# steps')
    axs[1].set_xlabel('Episode')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + '/train_log_moving_avg.png', dpi=150)
        plt.close()
    else:
        plt.show()


def plot_evaluation_log(env, max_steps_per_episode, data, save_path=None,
                        type='random', apply_scaling=False):
    """ Use rewards returned by evaluator function and create plots. """
    if apply_scaling:
        # Undo all scalings that have been applied to the reward and multiply by
        # 1'000 to get from [m] -> [mm]
        scaling = 1. / (env.state_scale * env.reward_scale) * 1000
    else:
        scaling = 1.

    # a) Evaluation log
    fig5, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot([(len(r) - 1) for r in data])
    axs[1].plot(np.array([r[0] for r in data]) * scaling, c='r', label='Initial')
    axs[1].plot(np.array([r[-1] for r in data]) * scaling, c='g', label='Final')
    try:
        axs[1].axhline(env.reward_threshold * scaling, color='k', ls='--')
    except AttributeError:
        axs[1].axhline(env.threshold * scaling, color='k', ls='--')

    axs[0].set_ylabel('Number of steps')
    axs[1].set_ylabel('Reward (mm)')
    axs[1].set_xlabel('Episodes')
    axs[1].legend(loc='lower left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + f'/eval_log_steps_rewards_{type}.png', dpi=150)
        plt.close()
    else:
        plt.show()

    # b) Extract and plot all intermediate rewards
    all_rewards = np.zeros((len(data), max_steps_per_episode + 1))
    cmap = plt.get_cmap("magma")
    fig6, axs = plt.subplots(2, 1, sharex=True)
    max_steps = 0
    for i in range(len(data)):
        all_rewards[i, :len(data[i])] = data[i]
        if len(data[i]) > max_steps:
            max_steps = len(data[i])
    axs[0].plot([(len(r) - 1) for r in data])
    for j in range(max_steps):
        axs[1].plot(all_rewards[:, j] * scaling, c=cmap(j / max_steps), alpha=0.7)
    axs[1].plot(np.array([r[0] for r in data]) * scaling, c='r', label='Initial')
    axs[1].plot(np.array([r[-1] for r in data]) * scaling, c='g', label='Final')
    try:
        axs[1].axhline(env.reward_threshold * scaling, color='k', ls='--')
    except AttributeError:
        axs[1].axhline(env.threshold * scaling, color='k', ls='--')

    axs[0].set_ylabel('Number of steps')
    axs[1].set_ylabel('Reward (mm)')
    axs[1].set_xlabel('Episodes')
    axs[1].legend(loc='lower left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + f'/eval_log_intermediate_rewards_{type}.png',
                    dpi=150)
        plt.close()
    else:
        plt.show()
