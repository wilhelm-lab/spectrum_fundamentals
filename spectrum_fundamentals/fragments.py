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


def compute_peptide_mass(sequence: str) -> float:
    """
    Compute the theoretical mass of the peptide sequence.

    :param sequence: Modified peptide sequence
    :return: Theoretical mass of the sequence
    """
    terminal_masses = 2 * constants.ATOM_MASSES["H"] + constants.ATOM_MASSES["O"]  # add terminal masses HO- and H-

    modification_deltas = _get_modifications(sequence)
    if modification_deltas:  # there were modifictions
        sequence = internal_without_mods([sequence])[0]
        terminal_masses += modification_deltas.get(-2, 0.0)  # prime with n_term_mod delta if present

    peptide_sum = sum([constants.AA_MASSES[c] + modification_deltas.get(i, 0.0) for i, c in enumerate(sequence)])

    return terminal_masses + peptide_sum


def _xl_sanity_check(noncl_xl: int, peptide_beta_mass: float, xl_pos: float):
    """
    Checks input validity for initialize_peacks when used with xl mode.

    :param noncl_xl: whether the function is called with a non-cleavable xl modification
    :param peptide_beta_mass: the mass of the second peptide to be considered for non-cleavable XL
    :param xl_pos: the position of the crosslinker for non-cleavable XL
    :raises ValueError: if non_cl_xl is 1 but no xl_pos and peptide_beta_mass has been supplied.
    """
    if noncl_xl == 1:
        if peptide_beta_mass == 0.0:
            raise ValueError("Peptide_beta_mass must be provided. Please check your input data.")
        if xl_pos == -1:
            raise ValueError("Crosslinker position must be provided if using non cleavable XL mode.")


def initialize_peaks(
    sequence: str,
    mass_analyzer: str,
    charge: int,
    mass_tolerance: Optional[float] = None,
    unit_mass_tolerance: Optional[str] = None,
    noncl_xl: bool = False,
    peptide_beta_mass: float = 0.0,
    xl_pos: int = -1,
) -> Tuple[List[dict], int, str, float]:
    """
    Generate theoretical peaks for a modified peptide sequence.

    :param sequence: Modified peptide sequence
    :param mass_analyzer: Type of mass analyzer used eg. FTMS, ITMS
    :param charge: Precursor charge
    :param mass_tolerance: mass tolerance to calculate min and max mass
    :param unit_mass_tolerance: unit for the mass tolerance (da or ppm)
    :param noncl_xl: whether the function is called with a non-cleavable xl modification
    :param peptide_beta_mass: the mass of the second peptide to be considered for non-cleavable XL
    :param xl_pos: the position of the crosslinker for non-cleavable XL
    :return: List of theoretical peaks, Flag to indicate if there is a tmt on n-terminus, Un modified peptide sequence
    """
    _xl_sanity_check(noncl_xl, peptide_beta_mass, xl_pos)

    max_charge = min(3, charge)
    ion_type_offsets = [0.0, constants.ATOM_MASSES["O"] + 2 * constants.ATOM_MASSES["H"]]

    # tmp place holder
    ion_type_masses = [0.0, 0.0]
    ion_types = ["b", "y"]

    number_of_ion_types = len(ion_type_offsets)
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

        ion_type_masses[0] = forward_sum + ion_type_offsets[0]  # b ion - ...

        ion_type_masses[1] = backward_sum + ion_type_offsets[1]  # y ion

        for charge in range(constants.MIN_CHARGE, max_charge + 1):  # generate ion in different charge states
            # positive charge is introduced by protons (or H - ELECTRON_MASS)
            charge_delta = charge * constants.PARTICLE_MASSES["PROTON"]
            for ion_type in range(number_of_ion_types):  # generate all ion types
                mass = _compute_ion_mass(
                    ion_mass=ion_type_masses[ion_type],
                    noncl_xl=noncl_xl,
                    ion_type=ion_type,
                    xl_pos=xl_pos,
                    peptide_beta_mass=peptide_beta_mass,
                    peptide_length=peptide_length,
                    i=i,
                )
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
    return fragments_meta_data, n_term_mod, sequence, (forward_sum + ion_type_offsets[0] + ion_type_offsets[1])


def _compute_ion_mass(
    ion_mass: float, noncl_xl: bool, ion_type: int, xl_pos: int, peptide_beta_mass: float, peptide_length: int, i: int
) -> float:
    # Check for neutral loss here

    if noncl_xl and ((ion_type == 0 and i + 1 >= xl_pos) or (ion_type == 1 and i >= peptide_length - xl_pos)):
        mass = ion_mass + peptide_beta_mass
    else:
        mass = ion_mass

    return mass


def initialize_peaks_xl(
    sequence: str,
    mass_analyzer: str,
    crosslinker_position: int,
    crosslinker_type: str,
    mass_tolerance: Optional[float] = None,
    unit_mass_tolerance: Optional[str] = None,
    sequence_beta: Optional[str] = None,
) -> Tuple[List[dict], int, str, float]:
    """
    Generate theoretical peaks for a modified (potentially cleavable cross-linked) peptide sequence.

    This function get only one modified peptide (peptide a or b))

    :param sequence: Modified peptide sequence (peptide a or b)
    :param mass_analyzer: Type of mass analyzer used eg. FTMS, ITMS
    :param crosslinker_position: The position of crosslinker
    :param crosslinker_type: Can be either DSSO, DSBU or BuUrBU
    :param mass_tolerance: mass tolerance to calculate min and max mass
    :param unit_mass_tolerance: unit for the mass tolerance (da or ppm)
    :param sequence_beta: optional second peptide to be considered for non-cleavable XL
    :raises ValueError: if crosslinker_type is unkown
    :raises AssertionError: if the short and long XL sequence (the one with the short / long crosslinker mod)
        has a tmt n term while the other one does not
    :return: List of theoretical peaks, flag to indicate if there is a tmt on n-terminus, unmodified peptide
        sequence, therotical mass of modified peptide (without considering mass of crosslinker)
    """
    crosslinker_type = crosslinker_type.upper()

    if crosslinker_type in ["DSSO", "DSBU", "BUURBU"]:  # cleavable XL
        charge = 2  # generate only peaks with charge 1 and 2
        if crosslinker_type == "DSSO":
            dsso = "[UNIMOD:1896]"
            dsso_s = "[UNIMOD:1881]"
            dsso_l = "[UNIMOD:1882]"
            sequence_s = sequence.replace(dsso, dsso_s)
            sequence_l = sequence.replace(dsso, dsso_l)
            sequence_without_crosslinker = sequence.replace(dsso, "")

        elif crosslinker_type in ["DSBU", "BUURBU"]:
            dsbu = "[UNIMOD:1884]"
            dsbu_s = "[UNIMOD:1886]"
            dsbu_l = "[UNIMOD:1885]"
            sequence_s = sequence.replace(dsbu, dsbu_s)
            sequence_l = sequence.replace(dsbu, dsbu_l)
            sequence_without_crosslinker = sequence.replace(dsbu, "")

        # TODO: this needs to be done more efficiently:
        # currently calculating the non-cleaved part until the xl_pos of the ions two times!
        # also the peptide sequence, mass and modifications are called twice
        # need to separate all of these functions better!
        # for XL, we actually don't need mass_s / mass_l at the moment because only one mass, without
        # the crosslinker is returned! This needs to be fixed, because mass is used as CALCULATED_MASS in
        # percolator!

        list_out_s, tmt_n_term_s, peptide_sequence, _ = initialize_peaks(
            sequence_s, mass_analyzer, charge, mass_tolerance, unit_mass_tolerance
        )
        list_out_l, tmt_n_term_l, peptide_sequence, _ = initialize_peaks(
            sequence_l, mass_analyzer, charge, mass_tolerance, unit_mass_tolerance
        )

        tmt_n_term = tmt_n_term_s
        if tmt_n_term_s ^ tmt_n_term_l:
            raise AssertionError("tmt_mod is {tmt_n_term_s} for short sequence but {tmt_n_term_l} for long sequence!")

        df_out_s = pd.DataFrame(list_out_s)
        df_out_l = pd.DataFrame(list_out_l)

        threshold_b = crosslinker_position
        threshold_y = len(peptide_sequence) - crosslinker_position + 1

        df_out_s.loc[(df_out_s["no"] >= threshold_b) & (df_out_s["ion_type"] == "b"), "ion_type"] = "b-short"
        df_out_s.loc[(df_out_s["no"] >= threshold_y) & (df_out_s["ion_type"] == "y"), "ion_type"] = "y-short"
        df_out_l.loc[(df_out_l["no"] >= threshold_b) & (df_out_l["ion_type"] == "b"), "ion_type"] = "b-long"
        df_out_l.loc[(df_out_l["no"] >= threshold_y) & (df_out_l["ion_type"] == "y"), "ion_type"] = "y-long"

        concatenated_df = pd.concat([df_out_s, df_out_l])
        unique_df = concatenated_df.drop_duplicates()  # TODO: this is where the duplicate calculations are removed
        df_out = unique_df.sort_values("mass")
        mass = compute_peptide_mass(sequence_without_crosslinker)

    elif crosslinker_type in ["BS3", "DSS"]:  # non-cleavable XL
        charge = 3  # generate only peaks with charge 1, 2 and 3

        sequence_without_crosslinker = sequence.replace("[UNIMOD:1898]", "")
        if sequence_beta is not None:
            sequence_beta_without_crosslinker = sequence_beta.replace("[UNIMOD:1898]", "")
        else:
            raise ValueError("sequence_beta cannot be None. Please check your input data.")
        sequence_mass = compute_peptide_mass(sequence_without_crosslinker)
        sequence_beta_mass = compute_peptide_mass(sequence_beta_without_crosslinker)

        list_out, tmt_n_term, peptide_sequence, _ = initialize_peaks(
            sequence,
            mass_analyzer,
            charge,
            mass_tolerance,
            unit_mass_tolerance,
            True,
            sequence_beta_mass if sequence_beta_mass is not None else None,
            crosslinker_position,
        )
        df_out = pd.DataFrame(list_out)

        threshold_b_alpha = crosslinker_position
        threshold_y_alpha = len(peptide_sequence) - crosslinker_position + 1

        df_out.loc[(df_out["no"] >= threshold_b_alpha) & (df_out["ion_type"] == "b"), "ion_type"] = "b-xl"
        df_out.loc[(df_out["no"] >= threshold_y_alpha) & (df_out["ion_type"] == "y"), "ion_type"] = "y-xl"

        df_out = df_out.sort_index()
        unique_df = df_out.drop_duplicates()
        df_out = unique_df.sort_values("mass")
        mass = sequence_mass

    else:
        raise ValueError(f"Unkown crosslinker type: {crosslinker_type}")

    return df_out.to_dict(orient="records"), tmt_n_term, peptide_sequence, mass


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


def compute_ion_masses(seq_int: List[int], charge_onehot: List[int], tmt: str = "") -> Optional[np.ndarray]:
    """
    Collects an integer sequence e.g. [1,2,3] with charge 2 and returns array with 174 positions for ion masses.

    Invalid masses are set to -1.

    :param seq_int: TODO
    :param charge_onehot: is a onehot representation of charge with 6 elems for charges 1 to 6
    :param tmt: TODO
    :return: list of masses as floats
    """
    charge = list(charge_onehot).index(1) + 1
    if not (charge in (1, 2, 3, 4, 5, 6) and len(charge_onehot) == 6):
        print("[ERROR] One-hot-enconded Charge is not in valid range 1 to 6")
        return None

    if not len(seq_int) == constants.SEQ_LEN:
        print(f"[ERROR] Sequence length {len(seq_int)} is not desired length of {constants.SEQ_LEN}")
        return None

    idx = list(seq_int).index(0) if 0 in seq_int else constants.SEQ_LEN
    masses = np.ones((constants.SEQ_LEN - 1) * 2 * 3, dtype=np.float32) * -1
    mass_b = 0
    mass_y = 0
    j = 0  # iterate over masses

    # Iterate over sequence, sequence should have length 30
    for i in range(idx - 1):  # only 29 possible ions
        j = i * 6  # index for masses array at position

        # MASS FOR Y IONS
        # print("Added", constants.VEC_MZ[seq_int[l-1-i]])
        mass_y += constants.VEC_MZ[seq_int[idx - 1 - i]]

        # Compute charge +1
        masses[j] = (
            mass_y
            + 1 * constants.PARTICLE_MASSES["PROTON"]
            + constants.MASSES["C_TERMINUS"]
            + constants.ATOM_MASSES["H"]
        ) / 1.0
        # Compute charge +2
        masses[j + 1] = (
            (
                mass_y
                + 2 * constants.PARTICLE_MASSES["PROTON"]
                + constants.MASSES["C_TERMINUS"]
                + constants.ATOM_MASSES["H"]
            )
            / 2.0
            if charge >= 2
            else -1.0
        )
        # Compute charge +3
        masses[j + 2] = (
            (
                mass_y
                + 3 * constants.PARTICLE_MASSES["PROTON"]
                + constants.MASSES["C_TERMINUS"]
                + constants.ATOM_MASSES["H"]
            )
            / 3.0
            if charge >= 3.0
            else -1.0
        )

        # MASS FOR B IONS
        if i == 0 and tmt != "":
            mass_b += constants.VEC_MZ[seq_int[i]] + constants.MOD_MASSES[constants.TMT_MODS[tmt]]
        else:
            mass_b += constants.VEC_MZ[seq_int[i]]

        # Compute charge +1
        masses[j + 3] = (
            mass_b
            + 1 * constants.PARTICLE_MASSES["PROTON"]
            + constants.MASSES["N_TERMINUS"]
            - constants.ATOM_MASSES["H"]
        ) / 1.0
        # Compute charge +2
        masses[j + 4] = (
            (
                mass_b
                + 2 * constants.PARTICLE_MASSES["PROTON"]
                + constants.MASSES["N_TERMINUS"]
                - constants.ATOM_MASSES["H"]
            )
            / 2.0
            if charge >= 2
            else -1.0
        )
        # Compute charge +3
        masses[j + 5] = (
            (
                mass_b
                + 3 * constants.PARTICLE_MASSES["PROTON"]
                + constants.MASSES["N_TERMINUS"]
                - constants.ATOM_MASSES["H"]
            )
            / 3.0
            if charge >= 3.0
            else -1.0
        )

    return masses
