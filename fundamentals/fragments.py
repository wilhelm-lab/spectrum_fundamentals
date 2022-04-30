import logging
import numpy as np
from operator import itemgetter

from . import constants as constants

logger = logging.getLogger(__name__)

def _get_modifications(peptide_sequence):
    """
    Get modification masses and position in a peptide sequence.
    :param peptide_sequence: Modified peptide sequence
    :return: Dict with modification position as an ID and mass as the value.
    """
    modification_deltas = {}
    tmt_n_term = 1
    modifications = constants.MOD_MASSES.keys()
    modification_mass = constants.MOD_MASSES
    # Handle terminal modifications here
    if peptide_sequence.startswith('[UNIMOD:737]'):  # TMT_6
        tmt_n_term = 2
        modification_deltas.update({0: constants.MOD_MASSES['[UNIMOD:737]']})
        peptide_sequence = peptide_sequence[12:]
    elif peptide_sequence.startswith('[UNIMOD:2016]'):  # TMT_16
        tmt_n_term = 2
        modification_deltas.update({0: constants.MOD_MASSES['[UNIMOD:2016]']})
        peptide_sequence = peptide_sequence[13:]
    elif peptide_sequence.startswith('[UNIMOD:214]'):  # iTRAQ4
        tmt_n_term = 2
        modification_deltas.update({0: constants.MOD_MASSES['[UNIMOD:214]']})
        peptide_sequence = peptide_sequence[12:]
    elif peptide_sequence.startswith('[UNIMOD:730]'):  # iTRAQ8
        tmt_n_term = 2
        modification_deltas.update({0: constants.MOD_MASSES['[UNIMOD:730]']})
        peptide_sequence = peptide_sequence[12:]
    elif peptide_sequence.startswith('[UNIMOD:1]'):  # acetylation
        modification_deltas.update({0: constants.MOD_MASSES['[UNIMOD:1]']})
        peptide_sequence = peptide_sequence[10:]

    if "(" in peptide_sequence:
        logger.info(
            'Error Modification ' + peptide_sequence[peptide_sequence.find('(') + 1:peptide_sequence.find(')')] + ' not '
                                                                                                                  'found')
        return

    while "[" in peptide_sequence:
        found_modification = False
        modification_index = peptide_sequence.index("[")
        for mod in modifications:
            if peptide_sequence[modification_index:modification_index + len(mod)] == mod:
                if modification_index - 1 in modification_deltas:
                    modification_deltas.update(
                        {modification_index - 1: modification_deltas[modification_index - 1] + modification_mass[
                            mod]})
                else:
                    modification_deltas.update({modification_index - 1: modification_mass[mod]})
                peptide_sequence = peptide_sequence[0:modification_index] + peptide_sequence[
                                                                            modification_index + len(mod):]
                found_modification = True
        if not found_modification:
            logger.info(
                'Error Modification ' + peptide_sequence[
                                        modification_index:peptide_sequence.find(']') + 1] + ' not found')
            return

    return modification_deltas, tmt_n_term, peptide_sequence


def _add_nl(neutral_losses, nl_dict, start_aa_index, end_aa_index):
    first_nl = True
    new_nls = {}
    for nl in neutral_losses:
        for i in range(start_aa_index, end_aa_index):
            current_aa_nl = nl_dict[i]
            if first_nl:
                new_nls[i] = list(set(neutral_losses) - set(current_aa_nl))
            if len(current_aa_nl) !=0:
                for exist_nl in current_aa_nl:
                    if exist_nl in new_nls[i]:
                        continue
                    if '-' not in exist_nl:
                        nl_list = [exist_nl, nl]
                        nl_list.sort()
                        nl_sorted = '-'.join(nl_list)
                        if nl_sorted in current_aa_nl:
                            continue
                        current_aa_nl.append(nl_sorted)
            if nl not in current_aa_nl:
                current_aa_nl.append(nl)
        first_nl = False
    return nl_dict


def _get_neutral_losses(peptide_sequence, modifications):
    """
    Get possible neutral losses and position in a peptide sequence.
    :param peptide_sequence: Modified peptide sequence
    :return: Dict with neutral losses position as an ID and composition as its value.
    """
    sequence_length = len(peptide_sequence)
    keys = range(0, sequence_length - 1)

    NL_b_ions = dict([(key, []) for key in keys])
    NL_y_ions = dict([(key, []) for key in keys])

    for i in range(0, sequence_length):
        aa = peptide_sequence[i]
        if aa in constants.AA_Neutral_losses:
            if i in modifications:
                if aa == 'M' and modifications[i] == 15.9949146:
                    NL_b_ions = _add_nl(constants.AA_Neutral_losses['M[UNIMOD:35]'], NL_b_ions, i, sequence_length - 1)
                    NL_y_ions = _add_nl(constants.AA_Neutral_losses['M[UNIMOD:35]'], NL_y_ions, sequence_length - i - 1,
                                        sequence_length - 1)
                elif aa == 'R' and modifications[i] == 0.984016:
                    NL_b_ions = _add_nl(constants.Mod_Neutral_losses['R[UNIMOD:7]'], NL_b_ions, i, sequence_length - 1)
                    NL_y_ions = _add_nl(constants.Mod_Neutral_losses['R[UNIMOD:7]'], NL_y_ions, sequence_length - i - 1,
                                        sequence_length - 1)
                else:
                    continue
            else:
                NL_b_ions = _add_nl(constants.AA_Neutral_losses[aa], NL_b_ions, i, sequence_length - 1)
                NL_y_ions = _add_nl(constants.AA_Neutral_losses[aa], NL_y_ions, sequence_length - i - 1, sequence_length - 1)
    return NL_b_ions, NL_y_ions

def _calculate_nl_score_mass(neutral_losses):
    score = 100
    mass = 0
    neutral_losses = neutral_losses.split('-')
    for nl in neutral_losses:
        mass += constants.Neutral_losses_Mass[nl]
        if nl == 'H2O' or nl == 'NH3':
            score -= 5
        else:
            score -= 30
    return score, mass

def compute_peptide_mass(sequence: str):
    """
    Compute the theoretical mass of the peptide sequence
    :param sequence: Modified peptide sequence.
    :return:Theoretical mass of the sequence
    """
    peptide_sequence = sequence
    modification_deltas, tmt_n_term, peptide_sequence = _get_modifications(peptide_sequence)

    peptide_length = len(peptide_sequence)
    if peptide_length > 30:
        return [], -1, ""

    n_term_delta = 0.0

    # get mass delta for the c-terminus
    c_term_delta = 0.0

    n_term = constants.ATOM_MASSES['H'] + n_term_delta  # n-terminal delta [N]
    c_term = constants.ATOM_MASSES['O'] + constants.ATOM_MASSES[
        'H'] + c_term_delta  # c-terminal delta [C]
    h = constants.ATOM_MASSES['H']

    ion_type_offsets = [n_term - h, c_term + h]

    # calculation:
    forward_sum = 0.0  # sum over all amino acids from left to right (neutral charge)

    for i in range(0, peptide_length):  # generate substrings
        forward_sum += constants.AA_MASSES[peptide_sequence[i]]  # sum left to right
        if i in modification_deltas:  # add mass of modification if present
            forward_sum += modification_deltas[i]
    return forward_sum+ion_type_offsets[0]+ion_type_offsets[1]


def initialize_peaks(sequence: str, mass_analyzer: str, charge: int):
    """
    Generate theoretical peaks for a modified peptide sequence.
    :param sequence: Modified peptide sequence.
    :param mass_analyzer: Type of mass analyzer used eg. FTMS, ITMS
    :param charge: Precursor charge
    :return: List of theoretical peaks, Flag to indicate if there is a tmt on n-terminus, Un modified peptide sequence
    """
    peptide_sequence = sequence
    modification_deltas, tmt_n_term, peptide_sequence = _get_modifications(peptide_sequence)
    nl_b_ions, nl_y_ions = _get_neutral_losses(peptide_sequence, modification_deltas)

    peptide_length = len(peptide_sequence)
    if peptide_length > 30:
        return [], -1, ""

    # initialize constants
    if int(round(charge)) <= 3:
        max_charge = int(round(charge))
    else:
        max_charge = 3

    n_term_delta = 0.0

    # get mass delta for the c-terminus
    c_term_delta = 0.0

    n_term = constants.ATOM_MASSES['H'] + n_term_delta  # n-terminal delta [N]
    c_term = constants.ATOM_MASSES['O'] + constants.ATOM_MASSES[
        'H'] + c_term_delta  # c-terminal delta [C]

    h = constants.ATOM_MASSES['H']

    ion_type_offsets = [n_term - h, c_term + h]

    # tmp place holder
    ion_type_masses = [0, 0]
    ion_types = ["b", "y"]

    number_of_ion_types = len(ion_type_offsets)
    fragments_meta_data = []
    # calculation:
    forward_sum = 0.0  # sum over all amino acids from left to right (neutral charge)
    backward_sum = 0.0  # sum over all amino acids from right to left (neutral charge)
    added_sequence = False
    for i in range(0, peptide_length - 1):  # generate substrings
        forward_sum += constants.AA_MASSES[peptide_sequence[i]]  # sum left to right
        if i in modification_deltas:  # add mass of modification if present
            forward_sum += modification_deltas[i]
        backward_sum += constants.AA_MASSES[peptide_sequence[peptide_length - i - 1]]  # sum right to left
        if peptide_length - i - 1 in modification_deltas:  # add mass of modification if present
            backward_sum += modification_deltas[peptide_length - i - 1]

        ion_type_masses[0] = forward_sum + ion_type_offsets[0]  # b ion - ...

        ion_type_masses[1] = backward_sum + ion_type_offsets[1]  # y ion

        for charge in range(constants.MIN_CHARGE, max_charge + 1):  # generate ion in different charge states
            # positive charge is introduced by protons (or H - ELECTRON_MASS)
            charge_delta = charge * constants.PARTICLE_MASSES["PROTON"]
            for ion_type in range(0, number_of_ion_types):  # generate all ion types
                # Check for neutral loss here
                mass = (ion_type_masses[ion_type] + charge_delta) / charge
                if mass_analyzer == 'FTMS':
                    min_mass = (mass * -20 / 1000000) + mass
                    max_mass = (mass * 20 / 1000000) + mass
                else:
                    min_mass = mass - 0.5
                    max_mass = mass + 0.5
                fragments_meta_data.append({'ion_type': ion_types[ion_type], 'no': i + 1, 'charge': charge,
                                            'mass': mass, 'min_mass': min_mass, 'max_mass': max_mass,
                                            'neutral_loss': '', 'score': 100-(charge-1)})
                possible_neutral_losses = []
                if i < peptide_length -1:
                    if ion_types[ion_type] == 'b':
                        possible_neutral_losses = nl_b_ions[i]
                    else:
                        possible_neutral_losses = nl_y_ions[i]
                for nl in possible_neutral_losses:
                    nl_score, nl_mass = _calculate_nl_score_mass(nl)

                    mass = (ion_type_masses[ion_type] - nl_mass + charge_delta) / charge
                    if mass_analyzer == 'FTMS':
                        min_mass = (mass * -20 / 1000000) + mass
                        max_mass = (mass * 20 / 1000000) + mass
                    else:
                        min_mass = mass - 0.5
                        max_mass = mass + 0.5
                    fragments_meta_data.append({'ion_type': ion_types[ion_type], 'no': i + 1, 'charge': charge,
                                                'mass': mass, 'min_mass': min_mass, 'max_mass': max_mass,
                                                'neutral_loss': nl, 'score': nl_score - (charge - 1)})

        fragments_meta_data = sorted(fragments_meta_data, key=itemgetter('mass'))
    return fragments_meta_data, tmt_n_term, peptide_sequence, (forward_sum+ion_type_offsets[0]+ion_type_offsets[1])


def compute_ion_masses(seq_int, charge_onehot, tmt=''):
    """
    Collects an integer sequence e.g. [1,2,3] with charge 2 and returns array with 174 positions for ion masses.
    Invalid masses are set to -1
    charge_one is a onehot representation of charge with 6 elems for charges 1 to 6
    """

    charge = list(charge_onehot).index(1) + 1
    if not (charge in (1, 2, 3, 4, 5, 6) and len(charge_onehot) == 6):
        print("[ERROR] One-hot-enconded Charge is not in valid range 1 to 6")
        return

    if not len(seq_int) == constants.SEQ_LEN:
        print("[ERROR] Sequence length {} is not desired length of {}".format(
            len(seq_int), constants.SEQ_LEN))
        return

    l = list(seq_int).index(0) if 0 in seq_int else constants.SEQ_LEN
    masses = np.ones((constants.SEQ_LEN-1)*2*3, dtype=np.float32)*-1
    mass_b = 0
    mass_y = 0
    j = 0  # iterate over masses

    # Iterate over sequence, sequence should have length 30
    for i in range(l-1):  # only 29 possible ions
        j = i*6  # index for masses array at position

        # MASS FOR Y IONS
        # print("Added", constants.VEC_MZ[seq_int[l-1-i]])
        mass_y += constants.VEC_MZ[seq_int[l-1-i]]

        # Compute charge +1
        masses[j] = (mass_y + 1*constants.PARTICLE_MASSES["PROTON"] +
                     constants.MASSES["C_TERMINUS"] + constants.ATOM_MASSES["H"])/1.0
        # Compute charge +2
        masses[j+1] = (mass_y + 2*constants.PARTICLE_MASSES["PROTON"] + constants.MASSES["C_TERMINUS"] +
                       constants.ATOM_MASSES["H"])/2.0 if charge >= 2 else -1.0
        # Compute charge +3
        masses[j+2] = (mass_y + 3*constants.PARTICLE_MASSES["PROTON"] + constants.MASSES["C_TERMINUS"] +
                       constants.ATOM_MASSES["H"])/3.0 if charge >= 3.0 else -1.0

        # MASS FOR B IONS
        if i == 0 and tmt == 'tmt':
            mass_b += constants.VEC_MZ[seq_int[i]]+229.162932
        elif i == 0 and tmt == 'tmtpro':
            mass_b += constants.VEC_MZ[seq_int[i]] + 304.207146
        elif i == 0 and tmt == 'itraq8':
            mass_b += constants.VEC_MZ[seq_int[i]] + 304.205360
        elif i == 0 and tmt == 'itraq4':
            mass_b += constants.VEC_MZ[seq_int[i]] + 144.102063
        else:
            mass_b += constants.VEC_MZ[seq_int[i]]

        # Compute charge +1
        masses[j+3] = (mass_b + 1*constants.PARTICLE_MASSES["PROTON"] +
                       constants.MASSES["N_TERMINUS"] - constants.ATOM_MASSES["H"])/1.0
        # Compute charge +2
        masses[j+4] = (mass_b + 2*constants.PARTICLE_MASSES["PROTON"] + constants.MASSES["N_TERMINUS"] -
                       constants.ATOM_MASSES["H"])/2.0 if charge >= 2 else -1.0
        # Compute charge +3
        masses[j+5] = (mass_b + 3*constants.PARTICLE_MASSES["PROTON"] + constants.MASSES["N_TERMINUS"] -
                       constants.ATOM_MASSES["H"])/3.0 if charge >= 3.0 else -1.0

    return masses
