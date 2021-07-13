AA_ALPHABET = {
    "A": 1,
    "C": 23,
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
}
ALPHABET_MODS = {
    "M(U:35)": 21,
    "C(U:4)": 2,  ## TODO check why this has the same value as C
    "K(U:737)": 22,
    "S(U:21)": 25,
    "T(U:21)": 26,
    "Y(U:21)": 27
}
ALPHABET = {**AA_ALPHABET, **ALPHABET_MODS}

MAXQUANT_VAR_MODS = {
    "M(ox)": "M(U:35)",
    "M(Oxidation (M))": "M(U:35)",
    "K(tm)": "K(U:737)",
    "S(ph)": "S(U:21)",
    "T(ph)": "T(U:21)",
    "Y(ph)": "Y(U:21)",
    "C(??)": "C(U:4)", # TODO Carbamidomethyl ##
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

MOD_MASSES = {
    'TMT_6': 229.162932,
    'Phospho': 79.966331,
    'Carbomedomethyl': 57.02146,
}

# initialize other masses
PROTON_MASS = 1.007276467
ELECTRON_MASS = 0.00054858

EPSILON = 1e-7

# masses of different atoms
ATOM_MASSES = {
    'H': 1.007825035,
    'C': 12.0,
    'O': 15.9949146,
    'N': 14.003074,
}

Meta_Data_Columns=['RAW_FILE',
                   'SCAN_NUMBER',
                   'MODIFIED_SEQUENCE',
                   'CHARGE',
                    'FRAGMENTATION',
                   'MASS_ANALYZER',
                   'MASS',
                   'SCORE',
                   'REVERSE']
