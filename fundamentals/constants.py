import numpy as np

ALPHABET = {
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    "M(ox)": 21,
    "K(tm)": 22,
}

AA_MASSES = {
    'A': 71.037114,
    'R': 156.101111,
    'N': 114.042927,
    'D': 115.026943,
    'C': 103.009185,
    'E': 129.042593,
    'Q': 128.058578,
    'G': 57.021464,
    'H': 137.058912,
    'I': 113.084064,
    'L': 113.084064,
    'K': 128.094963,
    'M': 131.040485,
    'F': 147.068414,
    'P': 97.052764,
    'S': 87.032028,
    'T': 101.047679,
    'U': 150.95363,
    'W': 186.079313,
    'Y': 163.063329,
    'V': 99.068414,
}

# initialize other masses
PROTON_MASS = 1.007276467
ELECTRON_MASS = 0.00054858

TMT_MASS = 229.162932

# masses of different atoms
ATOM_MASSES = {
    'H': 1.007825035,
    'C': 12.0,
    'O': 15.9949146,
    'N': 14.003074,
}

# small positive intensity to distinguish invalid ion (=0) from missing peak (=EPSILON)
EPSILON = np.nextafter(0, 1)

# peptide of length 30 has 29 b and y-ions, each with charge 1+, 2+ and 3+
NUM_IONS = 29*2*3

B_ION_MASK = np.tile([0,0,0,1,1,1], 29)
Y_ION_MASK = np.tile([1,1,1,0,0,0], 29)

