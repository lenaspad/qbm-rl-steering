import math
import random
import numpy as np
from typing import Dict, Tuple, Union
import gym

# from pyqubo import Spin
import dimod

# import matplotlib.pyplot as plt

# SQAOD (simulated quantum annealing)
try:
    from qbm_rl_steering.samplers.sqa_annealer import SQA
except ImportError:
    print('! Cannot import libraries (sqaod) required for SQA...')

# Amazon Braket
try:
    from qbm_rl_steering.samplers.qpu_annealer_dwave import QPU
except ImportError:
    print('! Cannot import libraries required for QPU (DWave)...')

# DWave SimulatedAnnealing
from qbm_rl_steering.samplers.sa_annealer import SA

# Qiskit samplers
try:
    from qbm_rl_steering.samplers.qaoa_solver import QAOA
except ImportError:
    print('! Cannot import libraries required for QAOA ...')


def get_visible_nodes_array(state: np.ndarray, action: np.ndarray,
                            state_space: gym.spaces.Box,
                            action_space: gym.spaces.Box) -> np.ndarray:
    """
    Normalize and concatenate state and action vectors to create state-action
    input vector (== "visible nodes" of clamped QBM).
    :param state: state as np.array from the environment (either through
    .reset(), or .step()).
    :param action: action as used in environment (continuous)
    :param state_space: openAI gym state space
    :param action_space: openAI gym action space
    :return normalized state-action vector
    """
    state_normalized = 2. * state / (state_space.high - state_space.low)
    action_normalized = 2. * action / (action_space.high - action_space.low)
    visible_nodes = np.array(list(state_normalized) + list(action_normalized))
    # print('visible_nodes', visible_nodes)
    return visible_nodes


def create_general_qubo_dict(
        w_hh: Dict, w_vh: Dict, visible_nodes: np.ndarray) -> Tuple[Dict, Dict]:
    """
    Creates dictionary of coupling weights of the graph. Corresponds to an
    upper triangular matrix, where self-coupling weights (linear coefficients)
    are on the diagonal, i.e. (i, i) keys, and the quadratic coefficients are
    on the off-diagonal, i.e. (i, j) keys with i < j. As the visible nodes
    are clamped, they are incorporated into biases, i.e. self-coupling
    weights of the hidden nodes they are connected to.
    :param w_hh: Contains key pairs (i, j) where i < j and the values
    are the coupling weights between the hidden nodes i and j.
    :param w_vh: Contains key pairs (visible, hidden) and the values are the
    coupling weights between visible and hidden nodes.
    :param visible_nodes: numpy array of inputs, i.e. visible nodes, given by
    the states and action vectors concatenated.
    :return Dictionary of the QUBO upper triangular Q-matrix that describes the
    quadratic equation to be minimized, as well as the original coefficients in
    Ising formulation.
    """
    qubo_dict = dict()

    # Add hidden-hidden coupling weights to QUBO matrix
    # (note that QUBO matrix is no longer made symmetrical to reduce the
    # number of coupling weights -> speeds up the sampling a bit)
    for k, w in w_hh.items():
        qubo_dict[k] = w

    # Self-coupling weights; clamp visible nodes to hidden nodes they are
    # connected to
    for k, w in w_vh.items():
        if (k[1], k[1]) not in qubo_dict:
            qubo_dict[(k[1], k[1])] = w * visible_nodes[k[0]]
        else:
            qubo_dict[(k[1], k[1])] += w * visible_nodes[k[0]]
    ising = qubo_dict.copy()

    J = {}
    h = {}

    # TODO: not so robust (nor elegant) maybe
    keylist = []
    for i, j in ising.keys():
        keylist.append(i)
        keylist.append(j)
    N = max(keylist) + 1
    for i in range(N):
        try:
            h[i] = ising[(i, i)]
        except KeyError:
            h[i] = 0

        for j in range(i + 1, N):
            try:
                J[i, j] = ising[(i, j)]
            except KeyError:
                J[i, j] = 0

    model = dimod.BinaryQuadraticModel(h, J, 0.0, vartype='SPIN')
    qubo, offset = model.to_qubo()

    return qubo, ising


def get_gradient_average_effective_hamiltonian(
        spin_configurations: np.ndarray, w_vh: Dict,
        visible_nodes: np.ndarray):
    """
    Calculate gradient with respect to visible nodes
    (note that we only take gradient of Heff wrt. v)
    """
    n_meas_for_average, n_replicas, _ = spin_configurations.shape

    h_derivative_wrt_v = np.zeros((len(visible_nodes), n_meas_for_average))
    for (v, h), w in w_vh.items():
        h_derivative_wrt_v[v, :] -= np.sum(
            w * spin_configurations[:, :, h], axis=-1) / n_replicas

    return np.float_(np.mean(h_derivative_wrt_v, axis=-1))


def get_average_effective_hamiltonian(
        spin_configurations: np.ndarray, w_hh: Dict, w_vh: Dict,
        visible_nodes: np.ndarray, big_gamma_final: float, beta_final: float)\
        -> float:
    """
    This method calculates the average effective Hamiltonian as given in
    Eq. (9) in paper: https://arxiv.org/pdf/1706.00074.pdf , using samples
    obtained from DWAVE QUBO sample method.
    :param spin_configurations: samples returned by the DWAVE sample() method,
    but converted to numpy array and reshaped to
    (n_meas_for_average, n_replicas, n_hidden_nodes). The samples contain the
    spin states (1 or -1) of all the hidden nodes.
    :param w_hh: dictionary of coupling weights (quadratic coefficients)
    between the hidden nodes of the Chimera graph. The key is (h, h') where h
    and h' are in range [0, .., n_hidden_nodes] and h != h' (no self-coupling
    here).
    :param w_vh: dictionary of coupling weights between the visible nodes
    (states-action vector) and the corresponding hidden nodes they are
    connected to. The key is (v, h), where v is the index of the visible node
    in range [0, .., n_bits_observation_space, .. , n_bits_observation_space
    + n_bits_action_space]. h is in range [0, .., n_hidden_nodes].
    :param visible_nodes: numpy array of inputs, i.e. visible nodes, given by
    binary vectors ({-1, +1}) of the concatenated states and action vectors
    (length: n_bits_observation_space + n_bits_action_space).
    :param big_gamma_final: final, i.e. at the end of the SQA, strength of
    the transverse field (virtual, average value), see paper for details.
    :param beta_final: Inverse temperature (note that this parameter is kept
    constant in SQA other than in SA).
    :return: single float; effective Hamiltonian averaged over the individual
    measurements.
    """
    _, n_replicas, _ = spin_configurations.shape

    # FIRST TERM, sum over h, h' in Eq. (9)
    h_sum_1 = 0.
    for (h, h_prime), w in w_hh.items():
        h_sum_1 += w * (spin_configurations[:, :, h] *
                        spin_configurations[:, :, h_prime])

    # SECOND term, sum over v, h in Eq. (9)
    h_sum_2 = 0.
    for (v, h), w in w_vh.items():
        h_sum_2 += w * visible_nodes[v] * spin_configurations[:, :, h]

    # Sum over replicas (sum over k), divide by n_replicas, and negate
    # This corresponds to the first line in Eq. (9)
    # h_sum_12 has shape (n_meas_for_average,)
    h_sum_12 = -np.sum(h_sum_1 + h_sum_2, axis=-1) / n_replicas

    # THIRD term, [-w_plus * (sum_hk_hkplus1 + sum_h1_hr)], in Eq. (9)
    # h_sum_3 has shape (n_meas_for_average,)
    if big_gamma_final == 0:
        # This is to remove the w_plus term in case we use classical SA.
        # TODO: is this correct?
        coth_term = 1.
    else:
        x = big_gamma_final * beta_final / n_replicas
        coth_term = math.cosh(x) / math.sinh(x)
    w_plus = math.log10(coth_term) / (2. * beta_final)

    # I think there is a typo in Eq. (9). The summation index in w+(.. + ..)
    # of the first term should only go from k=1 to r-1.
    hk_hkplus1_sum = np.sum(np.sum(
        spin_configurations[:, :-1, :] * spin_configurations[:, 1:, :],
        axis=1), axis=-1)
    h1_hr_sum = np.sum(
        spin_configurations[:, -1, :] * spin_configurations[:, 0, :],
        axis=-1)
    h_sum_3 = -w_plus * (hk_hkplus1_sum + h1_hr_sum)

    # Take average over n_meas_for_average
    # print('Avg. eff. Hamiltonian', float(np.mean(h_sum_12 + h_sum_3)))
    return float(np.mean(h_sum_12 + h_sum_3))


def get_free_energy(spin_configurations: np.ndarray, avg_eff_hamiltonian: float,
                    beta_final: float) -> float:
    """
    We count the number of unique spin configurations on the 3D extended
    Ising model (torus), i.e. on the n_replicas * n_hidden_nodes nodes and
    calculate the probability of occurrence for each spin configuration
    through mean values.
    :param spin_configurations: samples returned by the DWAVE sample() method,
    but converted to numpy array and reshaped to
    (n_meas_for_average, n_replicas, n_hidden_nodes). The samples contain the
    spin configurations (1 or -1) of all the hidden nodes (including the
    replicas, i.e. Trotter slices).
    :param avg_eff_hamiltonian: average effective Hamiltonian according to
    Eq. (9) in paper: https://arxiv.org/pdf/1706.00074.pdf .
    :param beta_final: Inverse temperature (note that this parameter is kept
    constant in SQA other than in SA).
    :return: free energy of the QBM defined according to the paper.
    """
    # Return the number of occurrences of unique spin configurations along
    # axis 0, i.e. along index of independent measurements
    _, n_occurrences = np.unique(
        spin_configurations, axis=0, return_counts=True)
    mean_n_occurrences = n_occurrences / float(np.sum(n_occurrences))
    a_sum = np.sum(mean_n_occurrences * np.log10(mean_n_occurrences))

    return avg_eff_hamiltonian + a_sum / beta_final


class QFunction(object):
    def __init__(self, sampler_type: str, state_space: gym.spaces.Box,
                 action_space: gym.spaces.Box, small_gamma: float,
                 n_replicas: int,
                 big_gamma: Union[Tuple[float, float], float],
                 beta: Union[float, Tuple[float, float]],
                 n_annealing_steps: int, n_meas_for_average: int,
                 kwargs_qpu, n_rows=2, n_columns=3, qfunc_it=0) -> None:
        """
        Implementation of the Q function (state-action value function).
        :param sampler_type: choose between simulated quantum annealing (SQA),
        classical annealing (SA), or Quantum annealing on hardware (QPU) (use
        big_gamma = 0 with SA)
        :param state_space: gym state space as initialized in the openAI gym
        environment (Box type).
        :param action_space: gym action space as initialized in the openAI gym
        environment (Box type).
        :param small_gamma: RL parameter, discount factor for
        cumulative future rewards.
        :param n_replicas: number of replicas (aka. Trotter slices) in the 3D
        extension of the Ising model, see Fig. 1 in paper:
        https://arxiv.org/pdf/1706.00074.pdf
        :param big_gamma: Transverse field; first entry is initial gamma and
        second entry is final gamma at end of annealing process (when using
        SQA). When annealer_type is SA, set big_gamma = 0.
        :param beta: Inverse temperature, either a float (for SQA),
        or a tuple of floats (for SA). For SQA, the temperature is kept
        constant.
        :param n_meas_for_average: number of times we run an independent
        annealing process from start to end
        :param n_annealing_steps: number of steps that one annealing
        process should take (~annealing time).
        :param kwargs_qpu: additional keyword arguments required for the
        initialization of the DWAVE QPU on Amazon Braket.
        """
        # TODO: adapt documentation
        # TODO: comment on that ... defines architecture of 'QPU'
        self.n_nodes_per_unit_cell = 8
        self.n_rows = n_rows  # 4
        self.n_columns = n_columns  # 4
        self.n_unit_cells = self.n_rows * self.n_columns
        n_graph_nodes = self.n_unit_cells * self.n_nodes_per_unit_cell

        if sampler_type == 'SQA':
            self.sampler = SQA(
                big_gamma=big_gamma, beta=beta, n_replicas=n_replicas,
                n_nodes=n_graph_nodes)
        elif sampler_type == 'SA':
            self.sampler = SA(
                beta=beta, big_gamma=big_gamma, n_replicas=n_replicas,
                n_nodes=n_graph_nodes, n_annealing_steps=n_annealing_steps)
        elif sampler_type == 'QPU':
            print('SETTING PROPER QPU AS SAMPLER')
            self.sampler = QPU(
                big_gamma=big_gamma, beta=beta, n_replicas=n_replicas, n_nodes=n_graph_nodes,
                qfunc_it=qfunc_it)
        elif sampler_type == 'QAOA':
            self.sampler = QAOA(n_nodes=n_graph_nodes, simulator='qasm',
                                n_shots=10, beta_final=beta)
        else:
            raise ValueError("sampler_type must be either 'SQA', 'SA', 'QPU', or 'QAOA'.")

        self.sampler_type = sampler_type

        self.n_annealing_steps = n_annealing_steps
        self.n_meas_for_average = n_meas_for_average
        self.n_replicas = n_replicas

        self.small_gamma = small_gamma

        # For normalization purposes
        self.state_space = state_space
        self.action_space = action_space

        self.w_hh, self.w_vh = self._initialise_weights()

        # Keep track of how weights evolve
        self.w_hh_history, self.w_vh_history = {}, {}
        for k in self.w_hh.keys():
            self.w_hh_history[k] = []
            self.w_hh_history[k].append(self.w_hh[k])
        for k in self.w_vh.keys():
            self.w_vh_history[k] = []
            self.w_vh_history[k].append(self.w_vh[k])

    def _initialise_weights(self, scale=1) -> Tuple[Dict, Dict]:
        """
        Initialise the coupling weights of the Chimera graph, i.e. both
        hidden-hidden couplings and visible-hidden couplings.
        """
        # ==============================
        # COUPLINGS BETWEEN HIDDEN NODES
        # This loop initializes hidden-hidden coupling weights to fully connect
        # nodes in a number of unit cells, e.g. of the Chimera graph (see Fig. 2
        # in paper: https://arxiv.org/pdf/1706.00074.pdf). The number of
        # unit cells can be arbitrary. The indexing of the nodes is starting
        # at the top left (node 0) and goes down vertically (blue nodes), and
        # then to the right (first red node is index 4). Note that inter-cell
        # couplings will be added in a separate loop below.
        w_hh = dict()
        for unit_idx in range(self.n_unit_cells):
            first_node = self.n_nodes_per_unit_cell * unit_idx
            for i in range(first_node, first_node + 4):
                for j in range(first_node + 4, first_node + 8):
                    w_hh[(i, j)] = (2 * random.random() - 1) * scale

        # This loop creates inter-cell couplings based on the chosen
        # architecture (i.e. n_rows vs. n_columns). Unit cells are horizontally
        # connected as 4 -> 12 -> 20 -> etc., 5 -> 13 -> 21 -> etc.,
        # while they are vertically connected as (if n_columns = 2) 0 -> 16
        # -> 32 -> etc., 1 -> 17 -> 33 -> etc.
        # 1) Horizontal inter-cell couplings
        for col in range(self.n_columns - 1):
            for row in range(self.n_rows):
                first_node = (col * self.n_nodes_per_unit_cell +
                              row * self.n_columns * self.n_nodes_per_unit_cell)
                for i, j in zip(tuple(range(first_node + 4, first_node + 8)),
                                tuple(range(first_node + 12, first_node + 16))):
                    w_hh[(i, j)] = (2 * random.random() - 1) * scale

        # 2) Vertical inter-cell couplings
        for row in range(self.n_rows - 1):
            for col in range(self.n_columns):
                first_node = (col * self.n_nodes_per_unit_cell +
                              row * self.n_columns * self.n_nodes_per_unit_cell)
                for i, j in zip(tuple(range(first_node, first_node + 4)),
                                tuple(range(
                                    first_node + 8 * self.n_columns,
                                    first_node + 8 * self.n_columns + 4))):
                    w_hh[(i, j)] = (2 * random.random() - 1) * scale

        # ==============================
        # COUPLINGS BETWEEN VISIBLE [the 'input' (= state layer) and the
        # 'output' (= action layer) of a 'classical' Q-net] AND THE HIDDEN NODES
        w_vh = dict()

        # Dense connection between the state nodes (visible) and the BLUE
        # hidden nodes (all 8 of them) of the Chimera graph. Blue nodes have
        # indices [0, 1, 2, 3, 12, 13, 14, 15, 16, 17, 18, 19, ...]. We hence
        # have connections between the state nodes [0, 1, ...,
        # len(state_space.high)] to all of the blue nodes.
        blue_nodes = tuple()
        for i in range(self.n_unit_cells):
            pick_col = i % 2
            first_node = i * self.n_nodes_per_unit_cell + pick_col * 4
            blue_nodes += tuple(range(first_node, first_node + 4))

        for j in blue_nodes:
            for i in range(len(self.state_space.high)):
                w_vh[(i, j)] = (2 * random.random() - 1) * scale

        # Dense connection between the action nodes (visible) and the RED hidden
        # nodes of the Chimera graph. Red nodes have indices [4, 5, 6, 7, 8,
        # 9, 10, 11, 20, 21, 22, 23, ...]. We hence have connections between the
        # action nodes [len(state_space.high), ..,
        # len(state_space.high) + len(action_space.high)] to all of the red
        # nodes.
        red_nodes = tuple()
        for i in range(self.n_unit_cells):
            pick_col = (i+1) % 2
            first_node = i * self.n_nodes_per_unit_cell + pick_col * 4
            red_nodes += tuple(range(first_node, first_node + 4))
        for j in red_nodes:
            for i in range(
                    len(self.state_space.high),
                    len(self.state_space.high) + len(self.action_space.high)):
                w_vh[(i, j)] = (2 * random.random() - 1) * scale

        return w_hh, w_vh

    # def draw_architecture(self):
    #     fig = plt.figure()
    #
    #     delta_x_intra = 2
    #     delta_x_inter = 4
    #     delta_y_intra = 2
    #     delta_y_inter = 4
    #
    #     # Nodes
    #     x_coords_unit_cell = np.array([0., delta_x_intra] * 4)
    #     y_coords_unit_cell = np.array([0., 0., -delta_y_intra, -delta_y_intra,
    #                                   -2*delta_y_intra, -2*delta_y_intra,
    #                                   -3*delta_y_intra, -3*delta_y_intra])
    #     for col in range(self.n_columns):
    #         for row in range(self.n_rows):
    #             plt.plot(x_coords_unit_cell, y_coords_unit_cell, 'ok')
    #
    #
    #     # Couplings
    #     for i, j in self.w_hh.keys():
    #         plt.plot()

    def calculate_q_value_on_batch(self, states, actions,
                                   calc_derivative: bool = False):
        q_values = []
        spin_configurations = []
        visible_nodes = []
        grads_wrt_s = []
        grads_wrt_a = []
        for i in range(len(states)):
            # print('calc q val on batch, states[i]', states[i])
            # print('calc q val on batch, actions[i]', actions[i])
            if i % 4 == 0:
                print(f'Sample in batch: {i + 1}/{len(states)}')

            if calc_derivative:
                q, sc, vn, grad_s, grad_a = self.calculate_q_value(
                    states[i], actions[i], calc_derivative=calc_derivative)
                q_values.append(q)
                spin_configurations.append(sc)
                visible_nodes.append(vn)
                grads_wrt_a.append(grad_a)
                grads_wrt_s.append(grad_s)
            else:
                q, sc, vn = self.calculate_q_value(
                    states[i], actions[i], calc_derivative=calc_derivative)
                q_values.append(q)
                spin_configurations.append(sc)
                visible_nodes.append(vn)
        q_values = np.array(q_values)
        spin_configurations = np.array(spin_configurations)
        visible_nodes = np.array(visible_nodes)

        if calc_derivative:
            grads_wrt_a = np.array(grads_wrt_a)
            grads_wrt_s = np.array(grads_wrt_s)
            return (q_values, spin_configurations, visible_nodes,
                    grads_wrt_s, grads_wrt_a)

        # print('q_values.shape', q_values.shape)
        return q_values, spin_configurations, visible_nodes

    def calculate_q_value(self, state: np.ndarray, action: np.ndarray,
                          calc_derivative: bool = False):
        """
        Based on state and chosen action, calculate the free energy,
        spin_configurations and vis_iterable.
        :param state: state the environment is in (binary vector, directly
        obtained from either env.reset(), or env.step())
        :param action: chosen action (continuous)
        :return free energy, spin_configurations, and visible_nodes.
        """
        # Define QUBO
        visible_nodes = get_visible_nodes_array(
            state=state, action=action,
            state_space=self.state_space, action_space=self.action_space)
        qubo_dict, ising_dict = create_general_qubo_dict(
            self.w_hh, self.w_vh, visible_nodes)

        # Run the annealing process (will be either SA, SQA, or QPU)
        spin_configurations = self.sampler.sample(
            qubo_dict=qubo_dict,
            n_meas_for_average=self.n_meas_for_average,
            n_steps=self.n_annealing_steps)

        # Based on sampled spin configurations calculate free energy
        avg_eff_hamiltonian = get_average_effective_hamiltonian(
            spin_configurations, self.w_hh, self.w_vh, visible_nodes,
            self.sampler.big_gamma_final, self.sampler.beta_final)

        free_energy = get_free_energy(
            spin_configurations, avg_eff_hamiltonian, self.sampler.beta_final)
        q_value = -free_energy
        # print('QUBO (q_value), proper formulation: ', q_value)

        # JUST FOR DEBUGGING, RUN SAME WITH ISING DICT
        # Run the annealing process (will be either SA, SQA, or QPU)
        # spin_configurations = self.sampler.sample(
        #     qubo_dict=ising_dict,
        #     n_meas_for_average=self.n_meas_for_average,
        #     n_steps=self.n_annealing_steps)
        #
        # # Based on sampled spin configurations calculate free energy
        # avg_eff_hamiltonian = get_average_effective_hamiltonian(
        #     spin_configurations, self.w_hh, self.w_vh, visible_nodes,
        #     self.sampler.big_gamma_final, self.sampler.beta_final)
        #
        # free_energy = get_free_energy(
        #     spin_configurations, avg_eff_hamiltonian, self.sampler.beta_final)
        # q_value_debug = -free_energy
        # print('ISING (q_value), improper form, old version', q_value_debug)

        if calc_derivative:
            grads_wrt_v = get_gradient_average_effective_hamiltonian(
                spin_configurations, self.w_vh, visible_nodes)
            grads_wrt_s = grads_wrt_v[:len(self.state_space.high)]
            grads_wrt_a = grads_wrt_v[len(self.state_space.high):]
            return (q_value, spin_configurations, visible_nodes, grads_wrt_s,
                    grads_wrt_a)

        return q_value, spin_configurations, visible_nodes

    def update_weights(
            self, spin_configurations: np.ndarray, visible_nodes: np.ndarray,
            current_q: float, future_q: float, reward: float,
            learning_rate: float, grad_clip: float = 1.) -> None:
        """
        Calculates the TD(0) learning step, i.e. the updates of the coupling
        dictionaries w_hh, w_vh according to Eqs. (11) and (12) in the paper:
        https://arxiv.org/pdf/1706.00074.pdf
        :param spin_configurations: spin configurations returned by the SQA.
        np array of shape (n_meas_for_average, n_replicas, n_hidden_nodes).
        The spin configurations contain the spin states (1 or -1) of all
        the hidden nodes (that includes replicas, i.e. Trotter slices).
        :param visible_nodes: numpy array of visible nodes, given by binary
        vectors (with -1 and +1) of the states and action vectors concatenated
        (length: n_bits_observation_space + n_bits_action_space).
        :param current_q: Q function value at time step n, Q(s_n, a_n)
        :param future_q: Q function value at time step n+1, Q(s_n+1, a_n+1)
        :param reward: RL reward of current step, r_n(s_n, a_n)
        :param learning_rate: RL. parameter, learning rate for update of
        coupling weights of the Chimera graph.
        :return None
        """
        # This term is the same for both weight updates w_hh and w_vh
        update_factor = learning_rate * (
                reward + self.small_gamma * future_q - current_q)

        # Update of w_vh, Eq. (11)
        h_avg = np.mean(np.mean(spin_configurations, axis=0), axis=0)
        for v, h in self.w_vh.keys():
            grads = update_factor * visible_nodes[v] * h_avg[h]
            grads = np.clip(grads, -grad_clip, grad_clip)
            self.w_vh[(v, h)] += grads

        # Update of w_hh, Eq. (12)
        for h, h_prime in self.w_hh.keys():
            grads = update_factor * np.mean(
                spin_configurations[:, :, h] *
                spin_configurations[:, :, h_prime])
            grads = np.clip(grads, -grad_clip, grad_clip)
            self.w_hh[(h, h_prime)] += grads

        # Keep track of the weights
        for k in self.w_hh.keys():
            self.w_hh_history[k].append(self.w_hh[k])
        for k in self.w_vh.keys():
            self.w_vh_history[k].append(self.w_vh[k])


# Derivative, numerical
# s = np.linspace(-1, 1, 20)
# a = np.linspace(-1, 1, 20)
# q = np.zeros((len(s), len(a)))
# dqda = np.zeros((len(s), len(a)))
# dqds = np.zeros((len(s), len(a)))
# for i, s_ in enumerate(s):
#     for j, a_ in enumerate(a):
#         a_ = np.atleast_1d(a_)
#         s_ = np.atleast_1d(s_)
#         q[i, j], _, _ = agent.critic.calculate_q_value(s_, a_)
#         dqda[i, j] = agent.get_action_derivative(s_, a_, epsilon=0.5)
#         dqds[i, j] = agent.get_state_derivative(s_, a_, epsilon=0.5)
# fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(8, 10))
# imq = axs[0].pcolormesh(s, a, q.T, shading='auto')
# fig.colorbar(imq, ax=axs[0])
# axs[0].set_title('Q')
# axs[0].set_ylabel('action')
#
# imdqda = axs[1].pcolormesh(s, a, dqda.T, shading='auto')
# axs[1].axvline(s[6], c='red')
# fig.colorbar(imdqda, ax=axs[1])
# axs[1].set_title('dq / da')
# axs[1].set_ylabel('action')
#
# imdqds = axs[2].pcolormesh(s, a, dqds.T, shading='auto')
# axs[2].axhline(a[5], c='red')
# fig.colorbar(imdqds, ax=axs[2])
# axs[2].set_title('dq / ds')
# axs[2].set_xlabel('state')
# axs[2].set_ylabel('action')
# plt.show()
#
# plt.figure()
# plt.suptitle('q vs dqda')
# plt.plot(a, q[6, :], label='Q')
# plt.plot(a, dqda[6, :], label='dQ/da')
# plt.legend()
# plt.xlabel('action')
# plt.ylabel('Q and dq/da resp.')
# plt.show()
#
# plt.figure()
# plt.suptitle('q vs dqds')
# plt.plot(s, q[:, 5], label='Q')
# plt.plot(s, dqds[:, 5], label='dQ/ds')
# plt.legend()
# plt.xlabel('state')
# plt.ylabel('Q and dq/ds resp.')
# plt.show()
