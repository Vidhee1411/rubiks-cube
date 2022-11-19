import numpy as np
import json
import pickle5 as pkl
import random

stateOrig = np.array(
    [-1, -1, 8, 1, 4, 7, 0, 3, 6, 11, -1, 17, 10, 13, 16, 9, 12, 15, 20, 23, 26, 19, 22, 25, 18, 21, 24, 29, 32, 35,
     28, 31, 34, 27, 30, 33, 42, 39, 36, -1, 40, 37, 44, -1, 38, 47, 50, 53, 46, 49, 52, 45, 48, 51]
)

with open('nlg_for_progress_file.pkl', 'rb') as f:
    nlg = pkl.load(f)

with open('display_data.pkl', 'rb') as f:
    var = pkl.load(f)

print("State:", var[7][25][1])


def getState(rand):
    state = var[rand][0]
    return np.array(state).tolist()


def getMoves():
    moves = var[rand][1]
    return moves.lower()
