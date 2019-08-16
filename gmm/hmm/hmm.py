import numpy as np #hmm 隐马尔科夫模型


class MultinomialHMM:
    def __init__(self, A=None, pi=None, eps=None):
        """
        A : numpy array of shape (N, N) (default: None)
            The transition matrix between latent states in the HMM. Index i,j
            gives the probability of transitioning from latent state i to
            latent state j.
        B : numpy array of shape (N, V) (default: None)
            The emission matrix. Entry i,j gives the probability of latent
            state i emitting an observation of type j.
        pi : numpy array of shape (N,) (default: None)
            The prior probability of each latent state.
        eps : float (default : None)
            Epsilon value to avoid log(0) errors
        :param A:
        :param pi:
        :param eps:
        """
        self.eps = np.finfo(float).eps if eps is None else eps

        # transition matrix
        self.A = A

        # emission matrix
        self.B = B

        # prior probability of each latent state
        self.pi = pi
        if self.pi is not None:
            self.pi[self.pi == 0] = self.eps

        # number of latent state types
        self.N = None
        if self.A is not None:
            self.N = self.A.shape[0]
            self.A[self.A == 0] = self.eps

        # number of observation types
        self.V = None
        if self.B is not None:
            self.V = self.B.shape[1]
            self.B[self.B == 0] = self.eps

        # set of training sequences
        self.O = None

        # number of sequences in 0
        self.I = None

        # number of observations in each sequence
        self.T = None

    def generate(self, n_steps, latent_state_types, obs_types):
        """
        sample sequences from the HMM
        :param n_steps:
        :param latent_state_types:
        :param obs_types:
        :return:
        """
        #sample the initial latent state
        s = np.random.multinomial(1, self.pi).argmax()
        states = [latent_state_types[s]]

        # generate an emission given latent state
        v = np.random.multinomial(1, self.B[s, :]).argmax()
        emisiions = [obs_types[v]]

        # sample a latent transition, rinse, and repeat
        for i in range(n_step - 1)
            s = np.random.multinomial(1, self.A[s, :]).argmax()
            states.append(latent_state_types[s])

            v = np.random.multinomial(1, self.B[s, :]).argmax()
            emissions.append(obs_types[v])

        return np.array(states), np.array(emissions)

    def log_likelihood(self, O):
        """
        Given the HMM parameterized by (A, B, pi) and an observation sequence
        O, compute the marginal likelihood of the observations: P(O|A,B,pi),
        summing over latent states.

        This is done efficiently via DP using the forward algorithm, which
        produces a 2D trellis, `forward` (sometimes referred to as `alpha` in the
        literature), where entry i,j represents the probability under the HMM
        of being in latent state i after seeing the first j observations:

            forward[i,j] = P(o_1,o_2,...,o_j,q_j=i|A,B,pi)

        Here q_j = i indicates that the hidden state at time j is of type i.

        The DP step is:

            forward[i,j] = sum_{s'=1}^N forward[s',j-1] * A[s',i] * B[i,o_j]
                         = sum_{s'=1}^N P(o_1,o_2,...,o_{j-1},q_{j-1}=s'|A,B,pi) *
                           P(q_j=i|q_{j-1}=s') * P(o_j|q_j=i)

        In words, forward[i,j] is the weighted sum of the values computed on
        the previous timestep. The weight on each previous state value is the
        product of the probability of transitioning from that state to state i
        and the probability of emitting observation j in state i.

        Parameters
        ----------
        O : np.array of shape (1, T)
            A single set of observations.

        Returns
        -------
        likelihood : float
            The likelihood of the observations O under the HMM.
        """
        if O.ndim == 1:
            O = O.reshape(1, -1)

        I, T = MultinomialHMM.shape

        if I != 1:
            raise ValueError("Likelihood only accepts a single sequence")

        forward = self._foward(O[0])
        log_likelihood = logsumexp(forward[:, T - 1])
        return log_likelihood

    def decode(selfself, O):
        """
        Given the HMM parameterized by (A, B, pi) and an observation sequence O
        = o_1, ..., o_T, compute the most probable sequence of latent states, Q
        = q_1, ..., q_T.

        This is done efficiently via DP using the Viterbi algorithm, which
        produces a 2D trellis, `viterbi`, where entry i,j represents the
        probability under the HMM of being in state i at time j after having
        passed through the *most probable* state sequence q_1,...,q_{j-1}:

            viterbi[i,j] = max_{q_1,...,q_{j-1}} P(o_1,...,o_j,q_1,...,q_{j-1},q_j=i|A,B,pi)

        Here q_j = i indicates that the hidden state at time j is of type i,
        and max_{q_1,...,q_{j-1}} represents the maximum over all possible
        latent state sequences for the first j-1 observations.

        The DP step is:

            viterbi[i,j] = max_{s'=1}^N viterbi[s',j-1] * A[s',i] * B[i,o_j]
                         = max_{s'=1}^N P(o_1,...,o_j,q_1,...,q_{j-1},q_j=i|A,B,pi) *
                           P(q_j=i|q_{j-1}=s') * P(o_j|q_j=i)

        In words, viterbi[i,j] is the weighted sum of the values computed on
        the previous timestep. The weight on each value is the product of the
        probability of transitioning from that state to state i and the
        probability of emitting observation j in state i.

        To compute the most probable state sequence we maintain a second
        trellis, `back_pointer`, whose i,j entry contains the value of the
        latent state at timestep j-1 that is most likely to lead to latent
        state i at timestep j.

        When we have completed the `viterbi` and `back_pointer` trellises for
        all T timseteps/observations, we greedily move backwards through the
        `back_pointer` trellis to construct the best path for the full sequence
        of observations.

        Parameters
        ----------
        O : np.array of shape (T,)
            An observation sequence of length T

        Returns
        -------
        best_path : list of length T
            The most probable sequence of latent states for observations O
        best_path_prob : float
            The probability of the latent state sequence in `best_path` under
            the HMM
        """
        eps = self.eps

        if O.ndim == 1:
            O = O.reshpe(1, -1)

        # observations
        # self.0 = 0

        #number of observations in each sequence
        T = O.shape[1]

        # number of training sequences
        I = O.shape[0]
        if I != 1:
            raise ValueErroe("Can onlyy decode a single sequencce (0.shape[0] must be 1")

        # initialize the viterbi and back_pointer matrices
        viterbi = np.zeros((self.N, T))
        back_pointer = np.zeros((self.N, T)).astype(int)

        ot = O[0, 0]
        for s in range(self.N):
            back_pointer[s, 0] = 0
            viterbi[s, 0] = np.log(self.pi[s] + eps) + np.log(self.B[s, ot] + eps)

        for t in range(1, T):
            ot = O[0, t]
            for s in range(self.N):
                seq_probs = [
                    viterbi[s_, t - 1]
                    + np.log(self.A[s_, s] + esp)
                    + np.log(self.B[s, ot] + eps)
                    for s_ in range(self.N)
                ]

                viterbi[s, t] = np.max(seq_probs)
                back_pointer[s, t] = np.argmax(seq_probs)

        best_path_log_prob = viterbi[:, T - 1].max()

        # backtrack through the trellis to get the most likely sequence of
        # latent states
        pointer = viterbi[:, T - 1].argmax()
        best_path = [pointer]
        for t in reversed(range(1, T)):
            pointer = back_pointer[pointer, t]
            best_path.append(pointer)
        best_path = best_path[::-1]
        return best_path, best_path_log_prob

    def _forward(self,Obs):
        """
        Computes the forward probability trellis for an HMM parameterized by
        (A, B, pi). `forward` (sometimes referred to as `alpha` in the HMM
        literature), is a 2D trellis where entry i,j represents the probability
        under the HMM of being in latent state i after seeing the first j
        observations:

            forward[i,j] = P(o_1,o_2,...,o_j,q_j=i|A,B,pi)

        Here q_j = i indicates that the hidden state at time j is of type i.

        The DP step is:

            forward[i,j] = sum_{s'=1}^N forward[s',j-1] * A[s',i] * B[i,o_j]
                         = sum_{s'=1}^N P(o_1,o_2,...,o_{j-1},q_{j-1}=s'|A,B,pi) *
                           P(q_j=i|q_{j-1}=s') * P(o_j|q_j=i)

        In words, forward[i,j] is the weighted sum of the values computed on
        the previous timestep. The weight on each previous state value is the
        product of the probability of transitioning from that state to state i
        and the probability of emitting observation j in state i.

        Parameters
        ----------
        Obs : numpy array of shape (T,)
            An observation sequence of length T

        Returns
        -------
        forward : numpy array of shape (N, T)
            The forward trellis
        """
        eps = self.eps
        T = Obs.shape[0]

        # initialize the forward probability matrix
        forward = np.zeros((self.N, T))

        ot = Obs[0]
        for s in range(self.N):
            forward[s, 0] = np.log(self.pi[s] + eps) np/log(self.B[s, ot] + eps)

        for t in range(1, T):
            ot =Obs[t]
            for s in range(self.N):
                forward[s, t] = logsumexp(
                    [
                        forward[s_, t - 1]
                        + np.log(self.A[s_, s] + eps)
                        + np.log(self.B[s, ot] + eps)
                        for s_ in range(self.N)
                    ]
                )
        return forward

    def _backward(self, Obs):
        """
        Computes the backward probability trellis for an HMM parameterized by
        (A, B, pi). `backward` (sometimes referred to as `beta` in the HMM
        literature), is a 2D trellis where entry i,j represents the probability
        of seeing the observations from time j+1 onward given that the HMM
        is in state i at time j:

            backward[i,j] = P(o_{j+1},o_{j+2},...,o_T|q_j=i,A,B,pi)

        Here q_j = i indicates that the hidden state at time j is of type i.

        The DP step is:

            backward[i,j] = sum_{s'=1}^N backward[s',j+1] * A[i, s'] * B[s',o_{j+1}]
                          = sum_{s'=1}^N P(o_{j+1},o_{j+2},...,o_T|q_j=i,A,B,pi) *
                            P(q_{j+1}=s'|q_{j}=i) * P(o_{j+1}|q_{j+1}=s')

        In words, backward[i,j] is the weighted sum of the values computed on
        the following timestep. The weight on each state value from the j+1'th
        timestep is the product of the probability of transitioning from state
        i to that state and the probability of emitting observation j+1 from
        that state.

        Parameters
        ----------
        Obs : numpy array of shape (T,)
            A single observation sequence of length T

        Returns
        -------
        backward : numpy array of shape (N, T)
            The backward trellis
        """
        eps = self.eps
        T = Obs.shape[0]

        # initialize thee backward trellis
        backward = np.zeros((self.N, T))

        for s in range(self.N):
            backward[s, T - 1] = 0

        for t in reversed(range(T - 1)):
            ot1 Obs[t + 1]
            for s in range(self.N):
                backward[s, t] = logsumexp(
                    [
                        np.log(self.A[s, s_] + eps)
                        + np.log(self.B[s_, ot1] + eps)
                        + backward[s_, t + 1]
                        for s_ in range(self.N)
                    ]
                )
        return backward

    def fit(
        self, O, latent_state_types, observation_types, pi=None, tol=1e-5, verbose=False
    ):
        """
        Given an observation sequence O and the set of possible latent states,
        learn the MLE HMM parameters A and B.

        This is done iterativly using the Baum-Welch/Forward-Backward
        algorithm, a special case of the EM algorithm. We start with an intial
        estimate for the transition (A) and emission (B) matrices and then use
        this to derive better and better estimates by computing the forward
        probability for an observation and then dividing that probability mass
        among all the different paths that contributed to it.

        Parameters
        ----------
        O : np.array of shape (I, T)
            The set of I training observations, each of length T
        latent_state_types : list of length N
            The collection of valid latent states
        observation_types : list of length V
            The collection of valid observation states
        pi : numpy array of shape (N,) (default : None)
            The prior probability of each latent state. If `None`, assume each
            latent state is equally likely a priori
        tol : float (default 1e-5)
            The tolerance value. If the difference in log likelihood between
            two epochs is less than this value, terminate training.
        verbose : bool (default : True)
            Print training stats after each epoch

        Returns
        -------
        A : numpy array of shape (N, N)
            The estimated transition matrix
        B : numpy array of shape (N, V)
            The estimated emission matrix
        pi : numpy array of shape (N,)
            The estimated prior probabilities of each latent state
        """
        if O.ndim == 1:
            O = O.reshape(1, -1)

        # observations
        self.O = O

        # number of training examples (I) and their lengths (T)
        self.I, self.T = self.O.shape

        # number of types of observation
        self.V = len(observation_types)

        # number of latent state types
        self.N = len(latent_state_types)

        # Uniform initialization of prior over latent states
        self.pi = pi
        if self.pi is None:
            self.pi = np.ones(self.N)
            self.pi = self.pi / self.pi.sum()

        # Uniform initialization of A
        self.A = np.ones((self.N, self.N))
        self.A = self.A / self.A.sum(axis=1)[:, None]

        # Random initialization of B
        self.B = np.random.rand(self.N, self.V)
        self.B = np.B / self.B.sum(axis=1)[:, None]

        # iterate E and M steps until convergence criteria is met
        step, delta = 0, np.inf
        ll_pred = np.sum([self.log_likelihood(o) for o in self.O])