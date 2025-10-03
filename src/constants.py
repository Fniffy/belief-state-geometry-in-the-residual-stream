"""
constants.py

This module defines application-wide constants used throughout the project.
"""

SHOW_VERSIONS = False

MAX_STACK_DEPTH = 15
MAX_STRING_LENGTH = 30
BATCH_SIZE = 128
MAX_CUT = 15
MAPPING = {'?' : 3, '(': 2, ')': 1, '_': 0}
EPOCHS = 10
LAYERS = 2
DIM_FEEDFORWARD = 64
DROPOUT = 0.1
TRAINING = False
EMBEDDING_DIMENSION = 64
CUT_PARENTHESES_MODEL = "belief-state-geometry-in-the-residual-stream/models/model_weights.pth"

if __name__ == "__main__":
    print(len(MAPPING))
