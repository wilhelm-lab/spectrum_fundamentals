import numpy as np
from enum import Enum

#####################
# GENERAL CONSTANTS #
#####################

SEQ_LEN = 30  # Sequence length for prosit
NUM_CHARGES_ONEHOT = 6
MAX_CHARGE = 6
BATCH_SIZE = 6000
VEC_LENGTH = 174

#############
# ALPHABETS #
#############

AA_ALPHABET = {
    "A": 1,
    "C": 24,
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

TERMINAL_ALPHABET = {
    "[]-": 30,  # unomodified n terminus
    "-[]": 31  # unomodified c terminus
}

ALPHABET_MODS = {
    "M[UNIMOD:35]": 21,
    "C[UNIMOD:4]": 2,
    "K[UNIMOD:737]": 22,
    "S[UNIMOD:21]": 25,
    "T[UNIMOD:21]": 26,
    "Y[UNIMOD:21]": 27,
    "[UNIMOD:1]-" : 32
}

ALPHABET = {**AA_ALPHABET, **ALPHABET_MODS}

######################
# MaxQuant constants #
######################

MAXQUANT_VAR_MODS = {
    "(ox)": "[UNIMOD:35]",
    "(Oxidation (M))": "[UNIMOD:35]",
    "(tm)": "[UNIMOD:737]",
    "(ph)": "[UNIMOD:21]",
    "C()": "C[UNIMOD:4]",  # TODO Investigate how MaxQuant encodes variable Carbamidomethyl
}

MAXQUANT_NC_TERM = {
    "^_": "",
    "_$": ""
}

####################
# MASS CALCULATION #
####################

# initialize other masses
PARTICLE_MASSES = {
    "PROTON": 1.007276467,
    "ELECTRON": 0.00054858
}

# masses of different atoms
ATOM_MASSES = {
    'H': 1.007825035,
    'C': 12.0,
    'O': 15.9949146,
    'N': 14.003074,
}

MASSES = {
        "PROTON": 1.007276467,
        "ELECTRON": 0.00054858,
        "H": 1.007825035,
        "C": 12.0,
        "O": 15.99491463,
        "N": 14.003074,
}

MASSES["N_TERMINUS"] = MASSES["H"]
MASSES["C_TERMINUS"] = MASSES["O"] + MASSES["H"]


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
    '[]-': MASSES["N_TERMINUS"],
    '-[]': MASSES["C_TERMINUS"],
}

MOD_MASSES = {
    '[UNIMOD:737]': 229.162932,  # TMT_6
    '[UNIMOD:21]': 79.966331,  # Phospho
    '[UNIMOD:4]': 57.02146,  # Carbamidomethyl
    '[UNIMOD:35]': 15.9949146  # Oxidation
}

AA_MOD_MASSES ={
    'K[UNIMOD:737]': 128.094963 + 229.162932
}

AA_MOD = {**AA_MASSES, **AA_MOD_MASSES}

#######################################
# HELPERS FOR FRAGMENT MZ CALCULATION #
#######################################

# Array containing masses --- at index one is mass for A, etc.
VEC_MZ = np.zeros(max(ALPHABET.values()) + 1)
for a, i in AA_ALPHABET.items():
    VEC_MZ[i] = AA_MOD[a]

# TODO Investigate where MOD_NAMES are used
MOD_NAMES = {
    '(U:737)': 'TMT_6',
    '(U:21)': 'Phospho',
    '(U:4)': 'Carbamidomethyl',
    '(U:35)': 'Oxidation'
}

# small positive intensity to distinguish invalid ion (=0) from missing peak (=EPSILON)
# EPSILON = 1e-7 # chec if it can be removed
EPSILON = 1e-7

# peptide of length 30 has 29 b and y-ions, each with charge 1+, 2+ and 3+
MAX_PEPTIDE_LEN = 30
NUM_IONS = (MAX_PEPTIDE_LEN - 1) * 2 * 3

B_ION_MASK = np.tile([0, 0, 0, 1, 1, 1], MAX_PEPTIDE_LEN - 1)
Y_ION_MASK = np.tile([1, 1, 1, 0, 0, 0], MAX_PEPTIDE_LEN - 1)

SHARED_DATA_COLUMNS = ['RAW_FILE', 'SCAN_NUMBER']
META_DATA_ONLY_COLUMNS = ['MODIFIED_SEQUENCE',
                          'PRECURSOR_CHARGE',
                          'FRAGMENTATION',
                          'MASS_ANALYZER',
                          'MASS',
                          'SCAN_EVENT_NUMBER',
                          'PRECURSOR_MASS_EXP',
                          'SCORE',
                          'REVERSE',
                          'RETENTION_TIME']
META_DATA_COLUMNS = SHARED_DATA_COLUMNS + META_DATA_ONLY_COLUMNS
MZML_ONLY_DATA_COLUMNS = ['INTENSITIES', 'MZ']
MZML_DATA_COLUMNS = SHARED_DATA_COLUMNS + MZML_ONLY_DATA_COLUMNS

SPECTRONAUT_MODS = {
    "M(U:35)": "oM"
}

FRAGMENTATION_ENCODING = {
    'HCD': 2,
    'CID': 1
}

############################
# GENERATION OF ANNOTATION #
############################

IONS = ['y', 'b']  # limited to single character unicode string when array is created
CHARGES = [1, 2, 3]  # limited to uint8 (0-255) when array is created
POSITIONS = [x for x in range(1, 30)]  # fragment numbers 1-29 -- limited to uint8 (0-255) when array is created

ANNOTATION_FRAGMENT_TYPE = []
ANNOTATION_FRAGMENT_CHARGE = []
ANNOTATION_FRAGMENT_NUMBER = []
for pos in POSITIONS:
    for ion in IONS:
        for charge in CHARGES:
            ANNOTATION_FRAGMENT_TYPE.append(ion)
            ANNOTATION_FRAGMENT_CHARGE.append(charge)
            ANNOTATION_FRAGMENT_NUMBER.append(pos)

ANNOTATION = [ANNOTATION_FRAGMENT_TYPE,
              ANNOTATION_FRAGMENT_CHARGE, ANNOTATION_FRAGMENT_NUMBER]


########################
# RESCORING PARAMETERS #
########################

class RESCORE_TYPE(Enum):
    PROSIT = 'prosit'
    ANDROMEDA = 'andromeda'

