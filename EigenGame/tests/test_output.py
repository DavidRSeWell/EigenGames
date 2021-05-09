import numpy as np

from ..data import MnistData
from ..eigengame import EigenGame


def test_ortho():
    """
    Test whether the output of a given run is
    returning vectors that are approximately orthogonal.
    Shouuld have a * b close to 0
    :return:
    """
    data = MnistData.load_with_sklearn()

    eigen_game = EigenGame(data)

    vectors = eigen_game.run()
    n = len(vectors)
    for i in range(n):
        v_i = vectors[i]
        for j in range(n):
            if i == j:
                continue

            v_j = vectors[j]

            d = np.dot(v_i.T,v_j)

            assert d < 0.00001


if __name__ == "__main__":
    test_ortho()