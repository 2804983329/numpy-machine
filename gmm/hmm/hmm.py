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
        self.0 = None

        # number of sequences in 0
        self.I = None

        # number of observations in each sequence
