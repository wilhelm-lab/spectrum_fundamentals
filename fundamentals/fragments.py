import logging
import numpy as np
from operator import itemgetter

from . import constants as constants

logger = logging.getLogger(__name__)
MIN_CHARGE = 1


def get_modifications(peptide_sequence):
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
    if peptide_sequence[:12] == '[UNIMOD:737]':  # TMT_6
        tmt_n_term = 2
        modification_deltas.update({0: constants.MOD_MASSES['[UNIMOD:737]']})
        peptide_sequence = peptide_sequence[12:]

    if "(" in peptide_sequence:
        logger.info(
            'Error Modification ' + peptide_sequence[peptide_sequence.find('(') + 1:peptide_sequence.find(')')] + 'not '
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


def initialize_peaks(sequence: str, mass_analyzer: str, charge: int):
    """
    Generate theoretical peaks for a modified peptide sequence.
    :param sequence: Modified peptide sequence.
    :param mass_analyzer: Type of mass analyzer used eg. FTMS, ITMS
    :param charge: Precursor charge
    :return: List of theoretical peaks, Flag to indicate if there is a tmt on n-terminus, Un modified peptide sequence
    """
    peptide_sequence = sequence
    modification_deltas, tmt_n_term, peptide_sequence = get_modifications(peptide_sequence)

    neutral_losses = []
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

    cho = constants.ATOM_MASSES['H'] + constants.ATOM_MASSES['C'] + constants.ATOM_MASSES[
        'O']
    h = constants.ATOM_MASSES['H']
    co = constants.ATOM_MASSES['C'] + constants.ATOM_MASSES['O']
    nh2 = constants.ATOM_MASSES['N'] + constants.ATOM_MASSES['H'] * 2.0

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
    for i in range(0, peptide_length):  # generate substrings
        forward_sum += constants.AA_MASSES[peptide_sequence[i]]  # sum left to right
        if i in modification_deltas:  # add mass of modification if present
            forward_sum += modification_deltas[i]
        backward_sum += constants.AA_MASSES[peptide_sequence[peptide_length - i - 1]]  # sum right to left
        if peptide_length - i - 1 in modification_deltas:  # add mass of modification if present
            backward_sum += modification_deltas[peptide_length - i - 1]

        ion_type_masses[0] = forward_sum + ion_type_offsets[0]  # b ion - ...

        ion_type_masses[1] = backward_sum + ion_type_offsets[1]  # y ion

        for charge in range(MIN_CHARGE, max_charge + 1):  # generate ion in different charge states
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
                                            'mass': mass, 'min_mass': min_mass, 'max_mass': max_mass})
        fragments_meta_data = sorted(fragments_meta_data, key=itemgetter('mass'))
    return fragments_meta_data, tmt_n_term, peptide_sequence, (forward_sum+ion_type_offsets[0]+ion_type_offsets[1])
