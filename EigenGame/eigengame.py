import numpy as np

from .data import Data

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def update_eigenvector(X: np.array, parents: list[np.array], d: int) -> tuple:
    """
    Follow Algorithm 1
    """
    v_i = sample_spherical(1, d)  # want d x 1
    lr = 0.001
    t_i = 1000  # TEMP
    # precompute rewards upfront for each vector of parents
    rewards_j = [np.matmul(X, v_j) for v_j in parents]
    for i in range(t_i):
        reward_i = np.matmul(X, v_i)  # n x 1

        penalty = np.zeros((reward_i.shape))
        for r_j in rewards_j:
            p = float(np.dot(reward_i.T, r_j) / np.dot(r_j.T, r_j))
            penalty += p * r_j

        delta_vi = 2.0 * np.matmul(X.T, (reward_i - penalty))

        reimann_projection = delta_vi - float(np.dot(delta_vi.T, v_i)) * v_i

        v_prime = v_i + lr * reimann_projection

        v_i /= np.linalg.norm(v_prime, axis=0)

    return v_i, parents


class EigenGame:

    def __init__(self,data: Data):
        """
        :param X:
        """
        self._X = data.X
        self._d = data.shape[1] # d = dimension of eigenvector. just assume that d = n for now

    def algorithm1(self) -> np.array:
        """
        Run algorithm1 from the paper
        :return:
        """
        vectors = []
        while(len(vectors) < self._d):
            v_i = self.update_eigenvector(vectors)
            vectors.append(v_i)

        return vectors

    def update_eigenvector(self,parents: list[np.array]) -> np.array:
        """
        Follow Algorithm 1
        """
        X = self._X
        d = self._d

        v_i = sample_spherical(1, d)  # want d x 1
        lr = 0.001
        t_i = 1000  # TEMP
        # precompute rewards upfront for each vector of parents
        rewards_j = [np.matmul(X, v_j) for v_j in parents]
        for i in range(t_i):
            reward_i = np.matmul(X, v_i)  # n x 1

            penalty = np.zeros((reward_i.shape))
            for r_j in rewards_j:
                p = float(np.dot(reward_i.T, r_j) / np.dot(r_j.T, r_j))
                penalty += p * r_j

            delta_vi = 2.0 * np.matmul(X.T, (reward_i - penalty))

            reimann_projection = delta_vi - float(np.dot(delta_vi.T, v_i)) * v_i

            v_prime = v_i + lr * reimann_projection

            v_i /= np.linalg.norm(v_prime, axis=0)

        return v_i

    def run(self,type="algo1") -> np.array:
        """
        Run the full algorithm. Return the principle components

        :param type: String that defines the algorithm being run
        :return:
        """

        if type == "algo1":
            return self.algorithm1()

