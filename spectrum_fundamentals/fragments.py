import itertools
import logging
import re
from operator import itemgetter
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

import spectrum_fundamentals.constants as c

from .mod_string import internal_without_mods

logger = logging.getLogger(__name__)


def _get_modifications(peptide_sequence: str, custom_mods: Optional[Dict[str, float]] = None) -> Dict[int, float]:
    """
    Get modification masses and position in a peptide sequence.

    This function expects a peptide sequence in unimod format, parses the modifications and stores
    the mass deltas for each position off aa in a dictionary where keys are the position in the
    unmodified sequence and values are the masses of the modifications attached to the aa at that
    position. In case of an n-terminal modification, it is stored at position -2 (technical reasons)
    The mass deltas along with the unmodified sequence and information about whether an n-terminal
    modification was present are returned.

    :param peptide_sequence: Modified peptide sequence
    :param custom_mods: mapping of custom UNIMOD string identifiers ('[UNIMOD:xyz]') to their mass
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

    mod_masses = c.MOD_MASSES | (custom_mods or {})

    for match in matches:
        start_pos, end_pos = match.span()
        modification_deltas[start_pos - offset] = mod_masses[match.group()]
        offset += end_pos - start_pos

    return modification_deltas


def compute_peptide_mass(sequence: str, custom_mods: Optional[Dict[str, float]] = None) -> float:
    """
    Compute the theoretical mass of the peptide sequence.

    :param sequence: Modified peptide sequence
    :param custom_mods: Custom Modifications with the identifier, the unimod equivalent and the respective mass
    :return: Theoretical mass of the sequence
    """
    terminal_masses = 2 * c.ATOM_MASSES["H"] + c.ATOM_MASSES["O"]  # add terminal masses HO- and H-

    modification_deltas = _get_modifications(sequence, custom_mods=custom_mods)
    if modification_deltas:  # there were modifictions
        sequence = internal_without_mods([sequence])[0]
        terminal_masses += modification_deltas.get(-2, 0.0)  # prime with n_term_mod delta if present

    peptide_sum = sum([c.AA_MASSES[aa] + modification_deltas.get(i, 0.0) for i, aa in enumerate(sequence)])

    return terminal_masses + peptide_sum


def _xl_sanity_check(noncl_xl: int, peptide_beta_mass: float, xl_pos: float):
    """
    Checks input validity for initialize_peaks when used with xl mode.

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


def retrieve_ion_types(fragmentation_method: str) -> List[str]:
    """
    Retrieve the ion types resulting from a fragmentation method in the correct order for dlomix predictions.

    Given the fragmentation method the function returns all ion types that can result from it.

    :param fragmentation_method: fragmentation method used during the MS
    :raises ValueError: if fragmentation_method is not supported
    :return: list of possible ion types
    """
    fragmentation_method = fragmentation_method.upper()
    ions = c.FRAGMENTATION_TO_IONS_BY_PAIRS.get(fragmentation_method, [])
    if not ions:
        raise ValueError(f"Unknown fragmentation method provided: {fragmentation_method}")
    return ions


def retrieve_ion_types_for_peak_initialization(fragmentation_method: str) -> List[str]:
    """
    Retrieve the ion types resulting from a fragmentation method in the correct order for peak initialization.

    Given the fragmentation method the function returns all ion types that can result from it.

    :param fragmentation_method: fragmentation method used during the MS
    :raises ValueError: if fragmentation_method is not supported
    :return: list of possible ion types
    """
    fragmentation_method = fragmentation_method.upper()
    ions = c.FRAGMENTATION_TO_IONS_BY_DIRECTION.get(fragmentation_method, [])
    if not ions:
        raise ValueError(f"Unknown fragmentation method provided: {fragmentation_method}")
    return ions


def get_ion_delta(ion_types: List[str]) -> np.ndarray:
    """
    Calculate the mass of an ion.

    :param ion_types: type of ions for which mass should be calculated
    :return: numpy array with masses of the ions
    """
    return np.array([c.ION_DELTAS[ion_type] for ion_type in ion_types]).reshape(len(ion_types), 1)


def _add_nl(neutral_losses: List[str], nl_dict: dict, start_aa_index: int, end_aa_index: int):
    """
    Adds neutral losses (NL) to a dictionary of neutral losses for specific amino acid indices.

    This function updates the `nl_dict` by incorporating the provided `neutral_losses` into
    the amino acid indices between `start_aa_index` and `end_aa_index`.

    :param neutral_losses: A list of neutral losses to be added to the amino acids.
    :param nl_dict: A dictionary where the keys are amino acid indices and the values are lists of neutral
        losses associated with each index.
    :param start_aa_index: The starting index of the amino acid range to which the neutral losses should be added.
    :param end_aa_index: The ending index of the amino acid range to which the neutral losses should be added.
    :returns: Updated dictionary with the added neutral losses for the specified amino acid indices.
    """
    first_nl = True
    new_nls = {}
    for nl in neutral_losses:
        for i in range(start_aa_index, end_aa_index):
            current_aa_nl = nl_dict[i]
            if first_nl:
                new_nls[i] = list(set(neutral_losses) - set(current_aa_nl))
            if nl not in current_aa_nl:
                current_aa_nl.append(nl)
        first_nl = False
    return nl_dict


def _get_neutral_losses(peptide_sequence, modifications):
    """
    Get possible neutral losses and position in a peptide sequence.

    :param peptide_sequence: Unmodified peptide sequence
    :param modifications: modifications dict generated by _get_modifications from modified petide sequence.
    :return: Dict with neutral losses position as an ID and composition as its value.
    """
    sequence_length = len(peptide_sequence)
    keys = range(0, sequence_length - 1)

    nl_b_ions = {key: [] for key in keys}
    nl_y_ions = {key: [] for key in keys}

    for i in range(0, sequence_length):
        aa = peptide_sequence[i]
        if aa in c.AA_Neutral_losses:
            if i in modifications:
                """if aa == "M" and modifications[i] == 15.9949146:
                nl_b_ions = _add_nl(c.AA_Neutral_losses["M[UNIMOD:35]"], nl_b_ions, i, sequence_length - 1)
                nl_y_ions = _add_nl(
                    c.AA_Neutral_losses["M[UNIMOD:35]"], nl_y_ions, sequence_length - i - 1, sequence_length - 1
                )"""
                if aa == "R" and modifications[i] == 0.984016:
                    nl_b_ions = _add_nl(c.Mod_Neutral_losses["R[UNIMOD:7]"], nl_b_ions, i, sequence_length - 1)
                    nl_y_ions = _add_nl(
                        c.Mod_Neutral_losses["R[UNIMOD:7]"], nl_y_ions, sequence_length - i - 1, sequence_length - 1
                    )
                elif (aa == "S" or aa == "T") and modifications[i] == 79.966331:
                    nl_b_ions = _add_nl(c.Mod_Neutral_losses["S[UNIMOD:21]"], nl_b_ions, i, sequence_length - 1)
                    nl_y_ions = _add_nl(
                        c.Mod_Neutral_losses["S[UNIMOD:21]"], nl_y_ions, sequence_length - i - 1, sequence_length - 1
                    )
            """else:
                nl_b_ions = _add_nl(c.AA_Neutral_losses[aa], nl_b_ions, i, sequence_length - 1)
                nl_y_ions = _add_nl(c.AA_Neutral_losses[aa], nl_y_ions, sequence_length - i - 1, sequence_length - 1)"""
    return nl_b_ions, nl_y_ions


def _calculate_nl_score_mass(neutral_loss):
    """
    Calculates the score and mass for a given neutral loss (NL).

    :param neutral_loss: The type of neutral loss for which to calculate the score and mass.
    :returns: A tuple containing the adjusted score and the mass of the specified neutral loss.
    """
    score = 100
    mass = 0
    mass = c.Neutral_losses_Mass[neutral_loss]
    if neutral_loss == "H2O" or neutral_loss == "NH3":
        score -= 5
    else:
        score -= 30
    return score, mass


def initialize_peaks(  # noqa: C901
    sequence: str,
    mass_analyzer: str,
    charge: int,
    mass_tolerance: Optional[float] = None,
    unit_mass_tolerance: Optional[str] = None,
    noncl_xl: bool = False,
    peptide_beta_mass: float = 0.0,
    xl_pos: int = -1,
    fragmentation_method: str = "HCD",
    custom_mods: Optional[Dict[str, float]] = None,
    add_neutral_losses: Optional[bool] = False,
) -> Tuple[List[dict], int, str, float, int]:
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
    :param fragmentation_method: fragmentation method that was used
    :param custom_mods: mapping of custom UNIMOD string identifiers ('[UNIMOD:xyz]') to their mass
    :param add_neutral_losses: Flag to indicate whether to annotate neutral losses or not
    :return: List of theoretical peaks, Flag to indicate if there is a tmt on n-terminus, Un modified peptide sequence,
        number of expected nl peaks
    """
    _xl_sanity_check(noncl_xl, peptide_beta_mass, xl_pos)

    max_charge = min(3, charge)
    ion_types = retrieve_ion_types_for_peak_initialization(fragmentation_method)
    modification_deltas = _get_modifications(sequence, custom_mods=custom_mods)

    fragments_meta_data = []
    n_term_mod = 1

    if noncl_xl:
        # the test only needs to be done because the unit tests use non_cl_xl peptides
        # without a crosslinker modification. This cannot occur in nature!!!
        # The unit tests need to be changed, then we can simply add to the existing
        # modification mass at xl_pos -1.
        modification_deltas[xl_pos - 1] = modification_deltas.get(xl_pos - 1, 0.0) + peptide_beta_mass

    if modification_deltas:  # there were modifictions
        sequence = internal_without_mods([sequence])[0]
        n_term_delta = modification_deltas.pop(-2, 0.0)  # directly pop it to avoid readding it later
        if n_term_delta != 0:
            n_term_mod = 2
            # add n_term mass to first aa for easy processing in the following calculation
            modification_deltas[0] = modification_deltas.get(0, 0.0) + n_term_delta

    if add_neutral_losses:
        nl_b_ions, nl_y_ions = _get_neutral_losses(sequence, modification_deltas)
        nl_ions = [nl_y_ions, nl_b_ions]

    expected_nl_count = 0
    mass_arr = np.array([c.AA_MASSES[_] for _ in sequence])
    for pos, mod_mass in modification_deltas.items():
        mass_arr[pos] += mod_mass

    n_forward_ions = len(ion_types) // 2
    n_fragments = len(sequence) - 1
    sum_array = np.empty(shape=(len(ion_types), n_fragments))
    np.cumsum(mass_arr[:0:-1], out=sum_array[0])  # this is for the reverse ion-series
    np.cumsum(mass_arr[:-1], out=sum_array[n_forward_ions])  # this is for the forward ion-series
    peptide_mass = sum_array[0, -1] + mass_arr[0]  # this is the longest reverse ion + the first residue

    # get offset for all needed ions
    deltas = get_ion_delta(ion_types)
    np.add(sum_array[0], deltas[:n_forward_ions], out=sum_array[:n_forward_ions])
    np.add(sum_array[n_forward_ions], deltas[n_forward_ions:], out=sum_array[n_forward_ions:])

    # calculate for m/z for charges 1, 2, 3
    # shape of ion_mzs: (n_ions, n_fragments, max_charge)
    charges = np.arange(1, max_charge + 1)
    ion_mzs = (sum_array[..., np.newaxis] + charges * c.PARTICLE_MASSES["PROTON"]) / charges
    min_mzs, max_mzs = get_min_max_mass(mass_analyzer, ion_mzs, mass_tolerance, unit_mass_tolerance)

    # write mz together with min and max value in output list with one dictionary for each ion
    for ion_type in range(len(ion_types)):
        for number in range(n_fragments):
            for charge in range(max_charge):
                fragments_meta_data.append(
                    {
                        "ion_type": ion_types[ion_type],  # ion type
                        "no": number + 1,  # no
                        "charge": charge + 1,  # charge
                        "mass": ion_mzs[ion_type, number, charge],  # mz
                        "min_mass": min_mzs[ion_type, number, charge],  # min mz
                        "max_mass": max_mzs[ion_type, number, charge],  # max mz
                        "neutral_loss": "",
                        "fragment_score": 100,
                    }
                )
                if not add_neutral_losses:
                    continue
                for nl in nl_ions[ion_type][number]:
                    nl_score, nl_mass = _calculate_nl_score_mass(nl)
                    ion_mass = sum_array[ion_type, number] - nl_mass
                    ion_mz = (ion_mass + (charge + 1) * c.PARTICLE_MASSES["PROTON"]) / (charge + 1)
                    min_mz, max_mz = get_min_max_mass(mass_analyzer, ion_mz, mass_tolerance, unit_mass_tolerance)
                    expected_nl_count += 1
                    fragments_meta_data.append(
                        {
                            "ion_type": ion_types[ion_type],  # ion type
                            "no": number + 1,  # no
                            "charge": charge + 1,  # charge
                            "mass": ion_mz,  # mz
                            "min_mass": min_mz,  # min mz
                            "max_mass": max_mz,  # max mz
                            "neutral_loss": nl,
                            "fragment_score": 100 - nl_score,
                        }
                    )

    fragments_meta_data = sorted(fragments_meta_data, key=itemgetter("mass"))

    return (
        fragments_meta_data,
        n_term_mod,
        sequence,
        (peptide_mass + c.ATOM_MASSES["O"] + 2 * c.ATOM_MASSES["H"]),
        expected_nl_count,
    )


def initialize_peaks_xl(
    sequence: str,
    mass_analyzer: str,
    crosslinker_position: int,
    crosslinker_type: str,
    mass_tolerance: Optional[float] = None,
    unit_mass_tolerance: Optional[str] = None,
    sequence_beta: Optional[str] = None,
    custom_mods: Optional[Dict[str, float]] = None,
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
    :param custom_mods: mapping of custom UNIMOD string identifiers ('[UNIMOD:xyz]') to their mass
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

        list_out_s, tmt_n_term_s, peptide_sequence, _, _ = initialize_peaks(
            sequence_s, mass_analyzer, charge, mass_tolerance, unit_mass_tolerance, custom_mods=custom_mods
        )
        list_out_l, tmt_n_term_l, peptide_sequence, _, _ = initialize_peaks(
            sequence_l, mass_analyzer, charge, mass_tolerance, unit_mass_tolerance, custom_mods=custom_mods
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

        list_out, tmt_n_term, peptide_sequence, _, _ = initialize_peaks(
            sequence,
            mass_analyzer,
            charge,
            mass_tolerance,
            unit_mass_tolerance,
            True,
            sequence_beta_mass if sequence_beta_mass is not None else None,
            crosslinker_position,
            custom_mods=custom_mods,
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
    mass_analyzer: str,
    mass: np.ndarray,
    mass_tolerance: Optional[float] = None,
    unit_mass_tolerance: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
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


FragmentIonComponent = Literal["ion_type", "position", "charge"]


def generate_fragment_ion_annotations(
    ion_types: List[str], order: Tuple[FragmentIonComponent, FragmentIonComponent, FragmentIonComponent]
) -> List[Tuple[str, int, int]]:
    """Generate full list of fragment ions for permitted ion types and specified order.

    :param ion_types: List of permitted ion types
    :param order: What fragment ion parameters (ion type, position & charge) to group the annotations by
    :return: List of (ion_type, position, charge) tuples sorted by specified component order
    :raises ValueError: if invalid or unsupported ion types are specified or duplicate order keys are used
    """
    fragment_ion_components: Dict[str, Union[List[str]]] = {
        "ion_type": ion_types,
        "position": [str(pos) for pos in c.POSITIONS],
        "charge": [str(charge) for charge in c.CHARGES],
    }

    if len(set(ion_types)) != len(ion_types):
        raise ValueError("Redundant ion types specified")
    elif len(ion_types) == 0:
        raise ValueError("No ion types specified")
    if set(order) != {"ion_type", "position", "charge"}:
        raise ValueError("Duplicate component used for ordering fragment ions")

    raw_annotations = list(itertools.product(*[fragment_ion_components[component] for component in order]))

    ordered_raw_annotations = [
        (
            str(combination[order.index("ion_type")]),
            int(combination[order.index("position")]),
            int(combination[order.index("charge")]),
        )
        for combination in raw_annotations
    ]

    return ordered_raw_annotations


def format_fragment_ion_annotation(raw_annotation: Tuple[str, int, int]) -> str:
    """Transform (ion_type, position, charge) tuple into <ion_type><position>+<charge> string.

    :param raw_annotation: `(ion_type, position, charge)` tuple
    :returns: formatted annotation string
    """
    ion_type, pos, charge = raw_annotation
    return f"{ion_type}{pos}+{charge}"
