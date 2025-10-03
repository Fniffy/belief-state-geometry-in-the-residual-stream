"""
constants.py

This module defines application-wide constants used throughout the project.
"""

SHOW_VERSIONS = False

MAX_STACK_DEPTH = 15
MAX_STRING_LENGTH = 20
BATCH_SIZE = 128
MAX_CUT = 15
MAPPING = {'?' : 3, '(': 2, ')': 1, '_': 0}
INV_MAPPING = {v: k for k, v in MAPPING.items()}
EPOCHS = 500
LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
TRAINING = True
EMBEDDING_DIMENSION = 128
CUT_PARENTHESES_MODEL = "belief-state-geometry-in-the-residual-stream/models/model_weights.pth"

if __name__ == "__main__":
    print(len(MAPPING))
