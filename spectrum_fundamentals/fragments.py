import logging
import re
from operator import itemgetter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import constants as constants
from .mod_string import internal_without_mods

logger = logging.getLogger(__name__)


def _get_modifications(peptide_sequence: str) -> Dict[int, float]:
    """
    Get modification masses and position in a peptide sequence.

    This function expects a peptide sequence in unimod format, parses the modifications and stores
    the mass deltas for each position off aa in a dictionary where keys are the position in the
    unmodified sequence and values are the masses of the modifications attached to the aa at that
    position. In case of an n-terminal modification, it is stored at position -2 (technical reasons)
    The mass deltas along with the unmodified sequence and information about whether an n-terminal
    modification was present are returned.

    :param peptide_sequence: Modified peptide sequence
    :return: modification_deltas
    """
    modification_deltas = {}
    offset = 1  # shift position of mod start in seq by one to the left to reflect position of aa
    if peptide_sequence.startswith("["):  # n-term mod => seq must be [UNIMOD:xyz]-X...
        offset = 2  # need to add one more offset, because of the dash '-', n_terminal stored at -1

    # fastest regex for modification mathing without lookback, since we know it must be unimod syntax
    # .{8} skips 8 positions entirely without checking greedily, that is len("UNIMOD:") + at least one digit
    # [^\]*] matches anything but ] greedily till it finds the closing bracket, which is 1 step
    pattern = re.compile(r"\[.{8}[^\]]*\]")
    matches = pattern.finditer(peptide_sequence)

    for match in matches:
        start_pos = match.start()
        end_pos = match.end()
        modification_deltas[start_pos - offset] = constants.MOD_MASSES[peptide_sequence[start_pos:end_pos]]
        offset += end_pos - start_pos

    return modification_deltas

def retrieve_ion_types(fragmentation_method: str) -> List[str]:
    """
    Retrieve the ion types resulting from a fragmentation method.

    Given the fragmentation method the function return all ion types that can result from it.

    : param fragmentation_method: fragmentation method used during the MS
    : return: list of the possible ion types
    """
    fragmentation_method = fragmentation_method.upper()
    if fragmentation_method == 'HCD' or fragmentation_method == 'CID':
        return ['b', 'y']
    elif fragmentation_method == 'ETD' or fragmentation_method == 'ECD':
        return ['c', 'z']
    elif fragmentation_method == 'ETCID' or fragmentation_method == 'ETHCD':
        return ['b', 'y', 'c', 'z']
    elif fragmentation_method == 'UVPD':
        return ['a', 'b', 'c', 'x', 'y', 'z']
    else:
        raise ValueError(f"Unknown fragmentation method provided: {fragmentation_method}")





def calculate_ion_mass(residual_mass: int, ion_type: str) -> int:
    """
    Calculate the mass of an ion.

    :param residual_mass: cumulative mass of the neutral residual masses
    :param ion_type: type of ion for which mass should be calculated
    :return mass of the ion

    """
    
    ion_type_offsets = {
            'a':  - constants.ATOM_MASSES["O"] - constants.ATOM_MASSES["C"], 
            'b': 0.0, 
            'c': 3 * constants.ATOM_MASSES["H"]+ constants.ATOM_MASSES["N"],
            'x': 2 * constants.ATOM_MASSES["O"] + constants.ATOM_MASSES["C"], 
            'y': constants.ATOM_MASSES["O"] + 2 * constants.ATOM_MASSES["H"],
            'z': constants.ATOM_MASSES["O"] - constants.ATOM_MASSES["N"] - constants.ATOM_MASSES["H"]
        }
    return residual_mass + ion_type_offsets[ion_type]




def initialize_peaks(
    sequence: str,
    mass_analyzer: str,
    charge: int,
    fragmentation_method: str,
    mass_tolerance: Optional[float] = None,
    unit_mass_tolerance: Optional[str] = None,
) -> Tuple[List[dict], int, str, float]:
    """
    Generate theoretical peaks for a modified peptide sequence.

    :param sequence: Modified peptide sequence
    :param mass_analyzer: Type of mass analyzer used eg. FTMS, ITMS
    :param charge: Precursor charge
    :param fragmentation_method 
    :param mass_tolerance: mass tolerance to calculate min and max mass
    :param unit_mass_tolerance: unit for the mass tolerance (da or ppm)
    :param noncl_xl: whether the function is called with a non-cleavable xl modification
    :param peptide_beta_mass: the mass of the second peptide to be considered for non-cleavable XL
    :param xl_pos: the position of the crosslinker for non-cleavable XL
    :return: List of theoretical peaks, Flag to indicate if there is a tmt on n-terminus, Un modified peptide sequence
    """
    
    max_charge = min(3, charge)
    
    # tmp place holder ???
    ion_types = retrieve_ion_types(fragmentation_method)
    print(ion_types)
    ion_type_masses = [0.0 for i in range (len(ion_types))]

    number_of_ion_types = len(ion_types)
    fragments_meta_data = []

    modification_deltas = _get_modifications(sequence)
    forward_sum = 0.0  # sum over all amino acids from left to right (neutral charge)
    backward_sum = 0.0  # sum over all amino acids from right to left (neutral charge)
    n_term_mod = 1
    if modification_deltas:  # there were modifictions
        sequence = internal_without_mods([sequence])[0]
        n_term_delta = modification_deltas.get(-2, 0.0)
        if n_term_delta != 0:
            n_term_mod = 2
            # add n_term mass to first aa for easy processing in the following calculation
            modification_deltas[0] = modification_deltas.get(0, 0.0) + n_term_delta

    # calculation:

    peptide_length = len(sequence)
    for i in range(peptide_length):  # generate substrings
        forward_sum += constants.AA_MASSES[sequence[i]]  # sum left to right
        if i in modification_deltas:  # add mass of modification if present
            forward_sum += modification_deltas[i]
        backward_sum += constants.AA_MASSES[sequence[peptide_length - i - 1]]  # sum right to left
        if peptide_length - i - 1 in modification_deltas:  # add mass of modification if present
            backward_sum += modification_deltas[peptide_length - i - 1]

        for j in range (len(ion_types)): # calculate masses of all ion types needed
            if ion_types[j] in ('a', 'b', 'c'):
                ion_type_masses[j] = calculate_ion_mass(forward_sum, ion_types[j]) 
            else:
                ion_type_masses[j] = calculate_ion_mass(backward_sum, ion_types[j])


        for charge in range(constants.MIN_CHARGE, max_charge + 1):  # generate ion in different charge states
            # positive charge is introduced by protons (or H - ELECTRON_MASS)
            charge_delta = charge * constants.PARTICLE_MASSES["PROTON"]
            for ion_type in range(number_of_ion_types):  # generate all ion types
                mass = ion_type_masses[ion_type]
                mz = (mass + charge_delta) / charge
                min_mz, max_mz = get_min_max_mass(mass_analyzer, mz, mass_tolerance, unit_mass_tolerance)
                fragments_meta_data.append(
                    {
                        "ion_type": ion_types[ion_type],  # ion type
                        "no": i + 1,  # no
                        "charge": charge,  # charge
                        "mass": mz,  # mz
                        "min_mass": min_mz,  # min mz
                        "max_mass": max_mz,  # max mz
                    }
                )
    fragments_meta_data = sorted(fragments_meta_data, key=itemgetter("mass"))
    return fragments_meta_data, n_term_mod, sequence, (forward_sum + constants.ATOM_MASSES["O"] + 2 * constants.ATOM_MASSES["H"])

def get_min_max_mass(
    mass_analyzer: str, mass: float, mass_tolerance: Optional[float] = None, unit_mass_tolerance: Optional[str] = None
) -> Tuple[float, float]:
    """Helper function to get min and max mass based on mass analyzer.

    If both mass_tolerance and unit_mass_tolerance are provided, the function uses the provided tolerance
    to calculate the min and max mass. If either `mass_tolerance` or `unit_mass_tolerance` is missing
    (or both are None), the function falls back to the default tolerances based on the `mass_analyzer`.

    Default mass tolerances for different mass analyzers:
    - FTMS: +/- 20 ppm
    - TOF: +/- 40 ppm
    - ITMS: +/- 0.35 daltons

    :param mass_tolerance: mass tolerance to calculate min and max mass
    :param unit_mass_tolerance: unit for the mass tolerance (da or ppm)
    :param mass_analyzer: the type of mass analyzer used to determine the tolerance.
    :param mass: the theoretical fragment mass
    :raises ValueError: if mass_analyzer is other than one of FTMS, TOF, ITMS
    :raises ValueError: if unit_mass_tolerance is other than one of ppm, da

    :return: a tuple (min, max) denoting the mass tolerance range.
    """
    if mass_tolerance is not None and unit_mass_tolerance is not None:
        if unit_mass_tolerance == "ppm":
            min_mass = (mass * -mass_tolerance / 1000000) + mass
            max_mass = (mass * mass_tolerance / 1000000) + mass
        elif unit_mass_tolerance == "da":
            min_mass = mass - mass_tolerance
            max_mass = mass + mass_tolerance
        else:
            raise ValueError(f"Unsupported unit for the mass tolerance: {unit_mass_tolerance}")
    elif mass_analyzer == "FTMS":
        min_mass = (mass * -20 / 1000000) + mass
        max_mass = (mass * 20 / 1000000) + mass
    elif mass_analyzer == "TOF":
        min_mass = (mass * -40 / 1000000) + mass
        max_mass = (mass * 40 / 1000000) + mass
    elif mass_analyzer == "ITMS":
        min_mass = mass - 0.35
        max_mass = mass + 0.35
    else:
        raise ValueError(f"Unsupported mass_analyzer: {mass_analyzer}")
    return (min_mass, max_mass)



