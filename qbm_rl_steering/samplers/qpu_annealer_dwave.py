from typing import Tuple, Dict
import numpy as np

try:
    from dwave.system import DWaveSampler
    from dwave.system import EmbeddingComposite
except ImportError:
    pass


class QPU:
    def __init__(self, big_gamma: Tuple[float, float], beta: float,
                 n_replicas: int, n_nodes: int = 72, qfunc_it: int = 0) -> None:
        """
        Initialize a hardware quantum annealer (QA) that runs on the DWAVE
        system. Note that we do not have much control over the annealing
        schedule for the hardware, so we assume that the given big_gammas are
        'fake' in a way, or need to be fitted (scanned), at least
        big_gamma_final.
        :param beta: inverse temperature (note that this parameter is kept
        constant in (S)QA other than in SA).
        :param n_replicas: number of replications of the graph in the extended
        dimension (number of Trotter slices).
        :param n_nodes: the number of qubits of the chip. This number is also
        the size of the square QUBO matrix that defines the couplings between
        the qubits / nodes.
        """
        self.n_nodes = n_nodes
        self.n_replicas = n_replicas
        self.n_calls = 0
        self.qfunc_it = qfunc_it

        # D-WAVE QA
        sampler = DWaveSampler(failover=True, retry_interval=10, solver='Advantage_system6.1') 
        print("QPU {} was selected.".format(sampler.solver.name))
        self.annealer = EmbeddingComposite(sampler)

        print(" ! Warning: big_gammas are 'virtual'. We don't know the actual "
              "values ... ")
        self.big_gamma_init = big_gamma[0]
        self.big_gamma_final = big_gamma[1]

        # Inverse temperature (assumed constant)
        if type(beta) is tuple:
            raise ValueError("When using QA, beta must be single float "
                             "(constant inverse temperature).")
        self.beta_init = beta
        self.beta_final = beta

    def sample(self, qubo_dict: Dict, n_meas_for_average: int,
               *args, **kwargs) -> np.ndarray:
        """
        Run the QPU DWAVE Sampler with the DWAVE QUBO method and generate all
        the spin configurations at hidden nodes of Chimera graph, with values
        {-1, +1}). The DWAVE sample() method provides samples in a list of
        dictionaries and the values are the corresponding spins {0, 1}. We
        will work with {-1, 1} rather than {0, 1}, so we remap all the
        sampled spin configurations and turn the list of dictionaries into a
        3D numpy array.
        :param qubo_dict: Dictionary of the coupling weights of the graph. It
        corresponds to an upper triangular matrix, where the self-coupling
        weights (linear coefficients) are on the diagonal, i.e. (i, i) keys,
        and the quadratic coefficients are on the off-diagonal, i.e. (i, j)
        keys with i < j. As the visible nodes are clamped, they are
        incorporated into biases, i.e. self-coupling weights of the hidden
        nodes they are connected to.
        :param n_meas_for_average: number of times we run an independent
        annealing process from start to end.
        :param: *args, **kwargs: accept these to match interface among
        annealers.
        :return Spin configurations, i.e. {-1, 1} of all the nodes and
        n_replicas (# Trotter slices) for all the n_meas_for_average runs we
        do. np array with dimensions
        (n_meas_for_average, n_replicas, n_hidden_nodes).
        """
        num_reads = n_meas_for_average * self.n_replicas

        spin_configurations = list(self.annealer.sample_qubo(
            Q=qubo_dict, num_reads=num_reads,
            # beta=self.beta_final,
            # postprocess='SAMPLING',
            # anneal_schedule=((0., 0.), (20., 1.)),
            # readout_thermalization=0,
            # reduce_intersample_correlation=True,
            answer_mode='raw'
        ).samples())

        # print(f"spin_configurations: {spin_configurations}")

        self.n_calls += 1
        # print(f"{self.qfunc_it}, N_CALLS: {self.n_calls}")
        # print(f"{self.qfunc_it}, N_READS: {num_reads}")

        # Convert to np array and flip all the 0s to -1s
        spin_configurations = np.array([
            list(s.values()) for s in spin_configurations])
        spin_configurations[spin_configurations == 0] = -1

        # Reshape
        spin_configurations = spin_configurations.reshape(
            (n_meas_for_average, self.n_replicas, self.n_nodes))

        return spin_configurations
