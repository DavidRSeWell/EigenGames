from abc import abstractmethod
from sklearn.datasets import load_digits
from sklearn.utils import Bunch


class Data:
    """
    Just mostly using for typing for now
    """

    @abstractmethod
    def __getitem__(self, i):
        pass

    @property
    @abstractmethod
    def shape(self):
        pass

    @property
    @abstractmethod
    def X(self):
        pass

    @property
    @abstractmethod
    def y(self):
        pass

class MnistData(Data):

    @classmethod
    def load_with_sklearn(cls):
        digits = load_digits()
        return cls(digits)

    def __init__(self,data_object: Bunch):
        """
        Sklearn likes to use Bunch (dictionary object) to store data. whatevs
        :param data_object: dictionary like object
        """
        self._data = data_object

    def __getitem__(self, i):
        return self.X[i]

    def __len__(self):
        return self._data.images.shape[0]

    @property
    def shape(self):
        return self.X.shape

    @property
    def X(self):
        return self._data.data
