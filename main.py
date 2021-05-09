import numpy as np

from EigenGame.data import MnistData
from EigenGame.eigengame import EigenGame

def main():

   data = MnistData.load_with_sklearn()

   eigen_game = EigenGame(data)

   vectors = eigen_game.run()

   print("Done running main")


if __name__ == "__main__":
   main()