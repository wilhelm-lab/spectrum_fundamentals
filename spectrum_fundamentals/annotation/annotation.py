import functools
import logging
import re
from operator import itemgetter
from typing import cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from spectrum_fundamentals import constants
from spectrum_fundamentals.fragments import (
    fragment_arrays,
    initialize_peaks,
    initialize_peaks_xl,
    retrieve_ion_types,
)

logger = logging.getLogger(__name__)


def match_peaks(  # noqa: C901
    fragments_meta_data: list[dict],
    peaks_intensity: np.ndarray,
    peaks_masses: np.ndarray,
    tmt_n_term: int,
    unmod_sequence: str,
    multifrag: bool = False,
    p_window_bounds: list | None = None,
) -> tuple[list[dict[str, str | int | float]], int, float]:
    """
    Matching experimental peaks with theoretical fragment ions.

    :param fragments_meta_data: Fragments ions meta data eg. ion type, number, theo_mass...
    :param peaks_intensity: Experimental peaks intensities
    :param peaks_masses: Experimental peaks masses
    :param tmt_n_term: Flag to check if there is tmt modification on n_terminus 1: no_tmt, 2:tmt
    :param unmod_sequence: Unmodified peptide sequence
    :param multifrag: Flag to check if it is a multifrag spectrum
    :param p_window_bounds: Peak exclusion window minimum and maximum boundaries
        for multifrag, used to remove precursor peaks (Da).
    :return: List of matched/annotated peaks
    """
    start_peak = 0
    no_of_peaks = len(peaks_intensity)
    max_intensity = 0.0
    row_list = []
    temp_list = []
    next_start_peak = 0
    matched_peak = False
    count_annotated_nl = 0
    fragment_no: float

    for fragment in fragments_meta_data:
        min_mass = fragment["min_mass"]
        max_mass = fragment["max_mass"]
        fragment_no = fragment["no"]
        if matched_peak:
            start_peak = next_start_peak
        matched_peak = False
        while start_peak < no_of_peaks:
            peak_mass = peaks_masses[start_peak]
            peak_intensity = peaks_intensity[start_peak]

            # precursor exclusion window
            if p_window_bounds is not None:
                if peak_mass > p_window_bounds[0] and peak_mass < p_window_bounds[1]:
                    start_peak += 1
                    continue

            if peak_mass > max_mass:
                break
            if peak_mass < min_mass:
                start_peak += 1
                continue

            if (
                not (
                    # unstable b1: if annotation is b1 and peptide sequence is not starting with R, H, K, S
                    (
                        (fragment["ion_type"] == "b" and fragment_no == 1)
                        and (not unmod_sequence.startswith(("R", "H", "K", "S")))
                    )
                    # unstable a2: if annotation is a2 and contains Q or N or has a Carbamidometylated C
                    or (
                        (fragment["ion_type"] == "a" and fragment_no == 2)
                        and (("Q" in unmod_sequence[:2]) or ("N" in unmod_sequence[:2]) or ("C" in unmod_sequence[:2]))
                    )
                )
                or (tmt_n_term == 2)
            ):
                # For now only counting neutral loss peaks this can change with different models later
                if fragment["neutral_loss"] == "":
                    meta_data = {
                        "ion_type": fragment["ion_type"],
                        "no": fragment_no,
                        "charge": fragment["charge"],
                        "exp_mass": peak_mass,
                        "theoretical_mass": fragment["mass"],
                        "intensity": peak_intensity,
                    }

                    if multifrag:
                        meta_data["full_name"] = fragment["full_name"]

                    row_list.append(meta_data)
                    if peak_intensity > max_intensity:
                        max_intensity = float(peak_intensity)
                else:
                    count_annotated_nl += 1

            matched_peak = True
            next_start_peak = start_peak
            start_peak += 1
    for row in row_list:
        row["intensity"] = float(row["intensity"]) / max_intensity
        temp_list.append(row)
    max_intensity_overall = max_intensity
    return temp_list, count_annotated_nl, max_intensity_overall


def handle_multiple_matches(
    matched_peaks: list[dict[str, str | int | float]], sort_by: str = "mass_diff"
) -> tuple[pd.DataFrame, int]:
    """
    Resolve cases where multiple peaks have been matched to the same fragment ion.

    This function takes a list of dictionaries representing matched peaks and resolves cases where multiple peaks have
    been matched to the same fragment ion. The function sorts the peaks based on the provided `sort_by` parameter and
    removes duplicate matches based on ion type, ion number, and charge state.

    :param matched_peaks: A list of dictionaries, each representing a matched peak. Each dictionary must contain the
                          following keys: 'ion_type', 'no', 'charge', 'exp_mass', 'theoretical_mass', and 'intensity'.
    :param sort_by: A string indicating the criterion to use when sorting matched peaks. Valid options are:
                    'mass_diff' (sort by absolute difference between experimental and theoretical mass,
                    ascending order),
                    'intensity' (sort by intensity, descending order), and 'exp_mass' (sort by experimental mass,
                    descending order).
    :raises ValueError: If an unsupported value is passed to `sort_by`.
    :return: A tuple containing a DataFrame of matched peaks (with duplicates removed) and an integer indicating the
             number of duplicate matches that were removed.
    """
    if sort_by == "mass_diff":
        matched_peaks = [
            peak | {"mass_diff": abs(cast(float, peak["exp_mass"]) - cast(float, peak["theoretical_mass"]))}
            for peak in matched_peaks
        ]

        def sort_key(peak):
            return peak["mass_diff"]

    elif sort_by == "intensity":

        def sort_key(peak):
            return -peak["intensity"]

    elif sort_by == "exp_mass":

        def sort_key(peak):
            return -peak["exp_mass"]

    else:
        raise ValueError(f"Unsupported value for sort_by supplied: {sort_by}")

    # best peak per fragment ion (ion type, number, charge); on ties the first one
    best: dict[tuple, tuple] = {}
    for i, peak in enumerate(matched_peaks):
        ion = (peak["ion_type"], peak["no"], peak["charge"])
        key = sort_key(peak)
        if ion not in best or key < best[ion][0]:
            best[ion] = (key, i, peak)
    kept = sorted(best.values(), key=itemgetter(0, 1))
    matched_peaks_df = pd.DataFrame([peak for _, _, peak in kept], index=[i for _, i, _ in kept])
    return matched_peaks_df, len(matched_peaks) - len(kept)


def annotate_spectra(
    un_annot_spectra: pd.DataFrame,
    mass_tolerance: float | None = None,
    unit_mass_tolerance: str | None = None,
    custom_mods: dict[str, float] | None = None,
    fragmentation_method: str = "HCD",
    multifrag: bool = False,
    p_window: float = 0.0,
    annotate_neutral_loss: bool = False,
    featured_ions: list[str] | None = None,
    phospho_loss_residues: str = "ST",
    plain_loss_residues: str | None = None,
) -> pd.DataFrame:
    """
    Annotate a set of spectra.

    This function takes a DataFrame of raw peaks and metadata, and for each spectrum, it calls the
    `parallel_annotate` function to annotate the spectrum and extract the necessary information.
    If there are any redundant peaks found in the annotation process, the function removes them and
    logs the information. Finally, it returns a Pandas DataFrame containing the annotated spectra
    with meta data.

    The returned DataFrame has the following columns:
    - INTENSITIES: a NumPy array containing the intensity values of each peak in the annotated spectrum
    - MZ: a NumPy array containing the m/z values of each peak in the annotated spectrum
    - CALCULATED_MASS: a float representing the calculated mass of the spectrum
    - removed_peaks: a NumPy array containing the indices of any peaks that were removed during the annotation process

    :param un_annot_spectra: a Pandas DataFrame containing the raw peaks and metadata to be annotated
    :param mass_tolerance: mass tolerance to calculate min and max mass
    :param unit_mass_tolerance: unit for the mass tolerance (da or ppm)
    :param fragmentation_method: fragmentation method that was used
    :param multifrag: flag to indicate whether to use multifrag or not
    :param p_window: peak exclusion window for multifrag, dedicated to remove precursor peaks (da)
    :param custom_mods: mapping of custom UNIMOD string identifiers ('[UNIMOD:xyz]') to their mass
    :param annotate_neutral_loss: flag to indicate whether to annotate neutral losses or not
    :param featured_ions: list of featured ions to annotate
    :param phospho_loss_residues: phosphorylated residues annotated with the loss of phosphoric acid
    :param plain_loss_residues: if given, the neutral losses are also counted with these residues instead of
        phospho_loss_residues (ANNOTATED_NL_COUNT_PLAIN, EXPECTED_NL_COUNT_PLAIN), e.g. "ST" if phospho_loss_residues
        also contains decoy residues
    :return: a Pandas DataFrame containing the annotated spectra with meta data
    """
    raw_file_annotations = []
    index_columns = {col: un_annot_spectra.columns.get_loc(col) for col in un_annot_spectra.columns}
    if not featured_ions:
        featured_ions = retrieve_ion_types(fragmentation_method)

    for row in tqdm(un_annot_spectra.values):
        results = parallel_annotate(
            row,
            index_columns,
            mass_tolerance,
            unit_mass_tolerance,
            fragmentation_method=fragmentation_method,
            multifrag=multifrag,
            p_window=p_window,
            custom_mods=custom_mods,
            annotate_neutral_losses=annotate_neutral_loss,
            featured_ions=featured_ions,
            phospho_loss_residues=phospho_loss_residues,
            plain_loss_residues=plain_loss_residues,
        )
        if not results:
            continue
        raw_file_annotations.append(results)
    results_df = pd.DataFrame(raw_file_annotations)

    if "CROSSLINKER_TYPE" not in index_columns:
        columns = [
            "INTENSITIES",
            "MZ",
            "CALCULATED_MASS",
            "removed_peaks",
            "ANNOTATED_NL_COUNT",
            "EXPECTED_NL_COUNT",
            "MOST_INTESE_PEAK",
        ]
        if plain_loss_residues is not None:
            columns += ["ANNOTATED_NL_COUNT_PLAIN", "EXPECTED_NL_COUNT_PLAIN"]
        results_df.columns = columns
    else:
        results_df.columns = [
            "INTENSITIES_A",
            "INTENSITIES_B",
            "MZ_A",
            "MZ_B",
            "CALCULATED_MASS_A",
            "CALCULATED_MASS_B",
            "removed_peaks_a",
            "removed_peaks_b",
        ]
    return results_df


def peak_pos_xl_cms2(unmod_seq: str, crosslinker_position: int) -> np.ndarray:
    """
    Determines the positions of all potential normal and xl fragments within the vector generated by
    ``generate_annotation_matrix``.

    This function is used only for cleavable crosslinked peptides.

    :param unmod_seq: Unmodified peptide sequence
    :param crosslinker_position: The position of the crosslinker
    :raises ValueError: if peptides exceed a length of 30
    :return: position of different fragments as list
    """
    xl_mask_array = np.full(constants.VEC_LENGTH_CMS2, -1.0)

    if len(unmod_seq) < constants.SEQ_LEN + 1:
        if crosslinker_position != 1:
            # b peaks
            peaks_b = np.tile([3, 4, 5], crosslinker_position - 1) + np.repeat(
                np.arange(crosslinker_position - 1) * 6, 3
            )
            xl_mask_array[peaks_b] = 0.0

            # ylong peaks
            first_pos_ylong = (len(unmod_seq) - crosslinker_position) * 6 + 174  # first position for ylong
            peaks_ylong = np.arange(first_pos_ylong, first_pos_ylong + 3)
            peaks_ylong = np.tile(peaks_ylong, crosslinker_position - 1)
            peaks_ylong += np.repeat(np.arange(crosslinker_position - 1) * 6, 3)

            xl_mask_array[peaks_ylong] = 0.0

            # yshort peaks
            xl_mask_array[[x - 174 for x in peaks_ylong]] = 0.0

        if len(unmod_seq) != crosslinker_position:
            # y peaks
            peaks_y = np.tile([0, 1, 2], len(unmod_seq) - crosslinker_position) + np.repeat(
                np.arange(len(unmod_seq) - crosslinker_position) * 6, 3
            )
            xl_mask_array[peaks_y] = 0.0

            # blong peaks
            first_pos_blong = (crosslinker_position - 1) * 6 + 174 + 3  # first position for blong
            peaks_blong = np.arange(first_pos_blong, first_pos_blong + 3)
            peaks_blong = np.tile(peaks_blong, len(unmod_seq) - crosslinker_position)
            peaks_blong += np.repeat(np.arange(len(unmod_seq) - crosslinker_position) * 6, 3)
            xl_mask_array[peaks_blong] = 0.0

            # bshort peaks
            xl_mask_array[[x - 174 for x in peaks_blong]] = 0.0

    else:
        raise ValueError(f"Peptides exceeding a length of 30 are not supported: {len(unmod_seq)}")

    return xl_mask_array


def generate_annotation_matrix_xl(
    matched_peaks: pd.DataFrame, unmod_seq: str, crosslinker_position: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the annotation matrix in the xl_prosit format from matched peaks.

    :param matched_peaks: matched peaks needed to be converted
    :param unmod_seq: unmodified peptide sequence
    :param crosslinker_position: position of crosslinker
    :return: numpy array of intensities and numpy array of masses
    """
    intensity = peak_pos_xl_cms2(unmod_seq, crosslinker_position)
    mass = intensity.copy()

    ion_type = matched_peaks.columns.get_loc("ion_type")
    no_col = matched_peaks.columns.get_loc("no")
    charge_col = matched_peaks.columns.get_loc("charge")
    intensity_col = matched_peaks.columns.get_loc("intensity")
    exp_mass_col = matched_peaks.columns.get_loc("exp_mass")

    for peak in matched_peaks.values:
        if peak[ion_type] == "y":
            peak_pos = ((peak[no_col] - 1) * 6) + (peak[charge_col] - 1)
        elif peak[ion_type] == "b":
            peak_pos = ((peak[no_col] - 1) * 6) + (peak[charge_col] - 1) + 3
        elif peak[ion_type] == "y-short":
            peak_pos = ((peak[no_col] - 1) * 6) + (peak[charge_col] - 1)
        elif peak[ion_type] == "b-short":
            peak_pos = ((peak[no_col] - 1) * 6) + (peak[charge_col] - 1) + 3
        elif peak[ion_type] == "y-long":
            peak_pos = ((peak[no_col] - 1) * 6) + (peak[charge_col] - 1) + 174
        else:
            peak_pos = ((peak[no_col] - 1) * 6) + (peak[charge_col] - 1) + 174 + 3

        if peak_pos >= constants.VEC_LENGTH_CMS2:
            continue
        intensity[peak_pos] = peak[intensity_col]
        mass[peak_pos] = peak[exp_mass_col]

    # convert elements representing charge 3 to -1 (we do not anotate +3)
    index_charge_3 = range(2, constants.VEC_LENGTH_CMS2, 3)
    intensity[index_charge_3] = -1
    mass[index_charge_3] = -1

    return intensity, mass


def generate_annotation_matrix(  # noqa: C901
    matched_peaks: pd.DataFrame,
    unmod_seq: str,
    charge: int,
    fragmentation_method: str = "HCD",
    multifrag: bool = False,
    featured_ions: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the annotation matrix in the prosit format from matched peaks.

    :param matched_peaks: matched peaks needed to be converted
    :param unmod_seq: Un modified peptide sequence
    :param charge: Precursor charge
    :param fragmentation_method: fragmentation method that was used
    :param multifrag: flag to indicate whether to annotate multifrag or not
    :param featured_ions: list of ions to be annotated
    :return: numpy array of intensities and numpy array of masses
    """
    if featured_ions is None:
        ion_types = retrieve_ion_types(fragmentation_method)
    else:
        ion_types = featured_ions

    if multifrag:
        ion_df = constants.ION_DIC.reset_index()
        vec_length = len(ion_df)  # length is 815
        ion_df = ion_df[ion_df["type"].isin(ion_types)]
        mask = (ion_df["num"] < len(unmod_seq) - 1) & (ion_df["charge"] <= charge)
        available_peaks = ion_df[mask].index.tolist()
        full_name_col = matched_peaks.columns.get_loc("full_name")
    else:
        ion_df = None
        charge_const = 3
        vec_length = (constants.SEQ_LEN - 1) * charge_const * len(ion_types)

        # change values to zeros
        if len(unmod_seq) < constants.SEQ_LEN:
            peaks_range = range(0, ((len(unmod_seq) - 1) * charge_const * len(ion_types)))
        else:
            peaks_range = range(0, ((constants.SEQ_LEN - 1) * charge_const * len(ion_types)))

        if charge == 1:
            available_peaks = [index for index in peaks_range if (index % 3 == 0)]
        elif charge == 2:
            available_peaks = [index for index in peaks_range if (index % 3 in (0, 1))]
        else:
            available_peaks = [index for index in peaks_range]

    intensity = np.full(vec_length, -1.0)
    mass = np.full(vec_length, -1.0)

    intensity[available_peaks] = 0.0
    mass[available_peaks] = 0.0

    ion_type = matched_peaks.columns.get_loc("ion_type")
    no_col = matched_peaks.columns.get_loc("no")
    charge_col = matched_peaks.columns.get_loc("charge")
    intensity_col = matched_peaks.columns.get_loc("intensity")
    exp_mass_col = matched_peaks.columns.get_loc("exp_mass")

    for peak in matched_peaks.values:
        if multifrag:
            try:
                # Skip redundant ions
                peak_pos = ion_df.index[ion_df["ion"] == peak[full_name_col]][0]
            except Exception:
                Warning(f"Peak {peak[full_name_col]} not found in ion_df")
                continue
        else:
            ion_type_index = ion_types.index(peak[ion_type].split("-", 1)[0])
            peak_pos = (
                ((peak[no_col] - 1) * charge_const * len(ion_types)) + (peak[charge_col] - 1) + 3 * ion_type_index
            )

            if peak_pos >= vec_length:
                continue
        intensity[peak_pos] = peak[intensity_col]
        mass[peak_pos] = peak[exp_mass_col]
    if (ion_df is None) and (len(unmod_seq) < constants.SEQ_LEN):
        mask_peaks = range(
            (len(unmod_seq) - 1) * charge_const * len(ion_types), ((len(unmod_seq)) * charge_const * len(ion_types))
        )
        intensity[mask_peaks] = -1.0
        mass[mask_peaks] = -1.0

    return intensity, mass


def parallel_annotate(
    spectrum: np.ndarray,
    index_columns: dict[str, int],
    mass_tolerance: float | None = None,
    unit_mass_tolerance: str | None = None,
    custom_mods: dict[str, float] | None = None,
    fragmentation_method: str = "HCD",
    multifrag: bool = False,
    featured_ions: list[str] | None = None,
    p_window: float = 0.0,
    annotate_neutral_losses: bool = False,
    phospho_loss_residues: str = "ST",
    plain_loss_residues: str | None = None,
) -> (
    tuple[np.ndarray, np.ndarray, float, int, int, int]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, int, int]
    | None
):
    """
    Perform parallel annotation of a spectrum.

    This function takes a spectrum and its index columns and performs parallel annotation of the spectrum.
    It starts by initializing the peaks and extracting necessary data from the spectrum.
    It then matches the peaks to the spectrum and generates an annotation matrix based on the matched peaks.
    If there are multiple matches found, it removes the redundant matches.
    Finally, it returns annotated spectrum with meta data including intensity values, masses, calculated
    masses, and any peaks that were removed. The function is designed to run in different threads to
    speed up the annotation pipeline.

    :param spectrum: a np.ndarray that contains the spectrum to be annotated
    :param index_columns: a dictionary that contains the index columns of the spectrum
    :param mass_tolerance: mass tolerance to calculate min and max mass
    :param unit_mass_tolerance: unit for the mass tolerance (da or ppm)
    :param custom_mods: mapping of custom UNIMOD string identifiers ('[UNIMOD:xyz]') to their mass
    :param fragmentation_method: fragmentation method that was used
    :param multifrag: flag to indicate whether to annotate multifrag or not
    :param featured_ions: list of ions to be annotated
    :param p_window: peak exclusion window for multifrag, dedicated to remove precursor peaks (da)
    :param annotate_neutral_losses: flag to indicate whether to annotate neutral losses or not
    :param phospho_loss_residues: phosphorylated residues annotated with the loss of phosphoric acid
    :param plain_loss_residues: if given, the neutral losses are also counted with these residues
    :return: a tuple containing intensity values (np.ndarray), masses (np.ndarray), calculated mass (float),
             and any removed peaks (List[str])
    """
    xl_type_col = index_columns.get("CROSSLINKER_TYPE")
    if xl_type_col is None:
        if spectrum[index_columns["PEPTIDE_LENGTH"]] > 30:  # this was in initialize peaks but can be checked prior
            return None
        return _annotate_linear_spectrum(
            spectrum,
            index_columns,
            mass_tolerance,
            unit_mass_tolerance,
            fragmentation_method=fragmentation_method,
            multifrag=multifrag,
            p_window=p_window,
            custom_mods=custom_mods,
            add_neutral_losses=annotate_neutral_losses,
            featured_ions=featured_ions,
            phospho_loss_residues=phospho_loss_residues,
            plain_loss_residues=plain_loss_residues,
        )

    if (spectrum[index_columns["PEPTIDE_LENGTH_A"]] > 30) or (spectrum[index_columns["PEPTIDE_LENGTH_B"]] > 30):
        return None
    return _annotate_crosslinked_spectrum(
        spectrum, index_columns, spectrum[xl_type_col], mass_tolerance, unit_mass_tolerance, custom_mods=custom_mods
    )


def _annotate_linear_spectrum(
    spectrum: np.ndarray,
    index_columns: dict[str, int],
    mass_tolerance: float | None,
    unit_mass_tolerance: str | None,
    custom_mods: dict[str, float] | None = None,
    fragmentation_method: str = "HCD",
    featured_ions: list[str] | None = None,
    multifrag: bool = False,
    p_window: float = 0.0,
    add_neutral_losses: bool = False,
    phospho_loss_residues: str = "ST",
    plain_loss_residues: str | None = None,
):
    """
    Annotate a linear peptide spectrum.

    :param spectrum: Spectrum to be annotated
    :param index_columns: Index columns of the spectrum
    :param mass_tolerance: Mass tolerance for calculating min and max mass
    :param unit_mass_tolerance: Unit for the mass tolerance (da or ppm)
    :param custom_mods: mapping of custom UNIMOD string identifiers ('[UNIMOD:xyz]') to their mass
    :param fragmentation_method: fragmentation method that was used
    :param add_neutral_losses: flag to indicate whether to annotate neutral losses or not
    :param multifrag: flag to indicate whether to annotate multifrag or not
    :param featured_ions: list of ions to be annotated
    :param p_window: peak exclusion window for multifrag, dedicated to remove precursor peaks (da)
    :param phospho_loss_residues: phosphorylated residues annotated with the loss of phosphoric acid
    :param plain_loss_residues: if given, the number of annotated and expected neutral losses with these residues
        instead of phospho_loss_residues are added to the result
    :return: Annotated spectrum
    """
    mod_seq_column = "MODIFIED_SEQUENCE"
    if "MODIFIED_SEQUENCE_MSA" in index_columns:
        mod_seq_column = "MODIFIED_SEQUENCE_MSA"

    if featured_ions is None:
        featured_ions = ["y", "b"]

    if _is_sorted(spectrum[index_columns["MZ"]]):
        return _annotate_linear_spectrum_arrays(
            spectrum,
            index_columns,
            mod_seq_column,
            mass_tolerance,
            unit_mass_tolerance,
            custom_mods,
            featured_ions,
            multifrag,
            p_window,
            add_neutral_losses,
            phospho_loss_residues,
            plain_loss_residues,
        )

    if multifrag:
        ion_df = constants.ION_DIC
        vec_length = len(ion_df)
    else:
        charge_const = 3
        vec_length = (constants.SEQ_LEN - 1) * charge_const * len(featured_ions)

    fragments_meta_data, tmt_n_term, unmod_sequence, calc_mass, expected_nl, p_window_bounds = initialize_peaks(
        sequence=spectrum[index_columns[mod_seq_column]],
        mass_analyzer=spectrum[index_columns["MASS_ANALYZER"]],
        charge=spectrum[index_columns["PRECURSOR_CHARGE"]],
        mass_tolerance=mass_tolerance,
        unit_mass_tolerance=unit_mass_tolerance,
        fragmentation_method=fragmentation_method,
        featured_ions=featured_ions,
        multifrag=multifrag,
        p_window=p_window,
        custom_mods=custom_mods,
        add_neutral_losses=add_neutral_losses,
        phospho_loss_residues=phospho_loss_residues,
    )

    matched_peaks, count_annotated_nl, max_annotated_intensity = match_peaks(
        fragments_meta_data,
        spectrum[index_columns["INTENSITIES"]],
        spectrum[index_columns["MZ"]],
        tmt_n_term,
        unmod_sequence,
        multifrag=multifrag,
        p_window_bounds=p_window_bounds,
    )

    plain_counts: tuple[int, ...] = ()
    if plain_loss_residues is not None:
        plain_counts = (count_annotated_nl, expected_nl)
        sequence = spectrum[index_columns[mod_seq_column]]
        extra_residues = "".join(set(phospho_loss_residues) - set(plain_loss_residues))
        if add_neutral_losses and extra_residues and re.search(rf"[{extra_residues}]\[UNIMOD:21\]", sequence):
            plain_fragments, _, _, _, plain_expected_nl, _ = initialize_peaks(
                sequence=sequence,
                mass_analyzer=spectrum[index_columns["MASS_ANALYZER"]],
                charge=spectrum[index_columns["PRECURSOR_CHARGE"]],
                mass_tolerance=mass_tolerance,
                unit_mass_tolerance=unit_mass_tolerance,
                fragmentation_method=fragmentation_method,
                featured_ions=featured_ions,
                multifrag=multifrag,
                p_window=p_window,
                custom_mods=custom_mods,
                add_neutral_losses=add_neutral_losses,
                phospho_loss_residues=plain_loss_residues,
            )
            _, plain_annotated_nl, _ = match_peaks(
                plain_fragments,
                spectrum[index_columns["INTENSITIES"]],
                spectrum[index_columns["MZ"]],
                tmt_n_term,
                unmod_sequence,
                multifrag=multifrag,
                p_window_bounds=p_window_bounds,
            )
            plain_counts = (plain_annotated_nl, plain_expected_nl)

    if len(matched_peaks) == 0:
        intensity = np.full(vec_length, 0.0)
        mass = np.full(vec_length, 0.0)
        return (intensity, mass, calc_mass, 0, 0, 0, np.nan) + ((0, 0) if plain_counts else ())

    matched_peaks, removed_peaks = handle_multiple_matches(matched_peaks)

    intensities, mass = generate_annotation_matrix(
        matched_peaks,
        unmod_sequence,
        spectrum[index_columns["PRECURSOR_CHARGE"]],
        fragmentation_method,
        multifrag=multifrag,
        featured_ions=featured_ions,
    )

    return (
        intensities,
        mass,
        calc_mass,
        removed_peaks,
        count_annotated_nl,
        expected_nl,
        max_annotated_intensity,
    ) + plain_counts


def _is_sorted(values: np.ndarray) -> bool:
    """Check that the peak m/z are in ascending order (assumed by the scan in match_peaks)."""
    values = np.asarray(values)
    return bool(np.all(values[1:] >= values[:-1]))


@functools.lru_cache(maxsize=1024)
def _empty_annotation(seq_len: int, charge: int, ion_types: tuple[str, ...], multifrag: bool) -> np.ndarray:
    """
    Annotation vector before peaks are filled in, as in generate_annotation_matrix.

    0 for the fragments that can exist (fragment number below the sequence length, charge up to the precursor charge),
    -1 otherwise. Cached, callers must copy it.

    :param seq_len: length of the unmodified sequence
    :param charge: precursor charge (capped at 3 without multifrag)
    :param ion_types: ion types
    :param multifrag: vector of the multifrag ion dictionary (constants.ION_DIC)
    :return: annotation vector
    """
    if multifrag:
        ion_df = constants.ION_DIC
        vector = np.full(len(ion_df), -1.0)
        available = ion_df["type"].isin(ion_types) & (ion_df["num"] < seq_len - 1) & (ion_df["charge"] <= charge)
        vector[available.to_numpy()] = 0.0
    else:
        charge_const = 3
        vector = np.full((constants.SEQ_LEN - 1) * charge_const * len(ion_types), -1.0)
        n_available = (min(seq_len, constants.SEQ_LEN) - 1) * charge_const * len(ion_types)
        vector[:n_available][np.arange(n_available) % charge_const < charge] = 0.0
    vector.flags.writeable = False
    return vector


def _match_fragments(
    fragments: dict[str, np.ndarray],
    peaks_masses: np.ndarray,
    tmt_n_term: int,
    unmod_sequence: str,
    featured_ions: list[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Match peaks to fragments, vectorized version of match_peaks for the arrays of fragment_arrays.

    Every peak inside the window of a fragment is matched. Like match_peaks, the scan of a fragment starts at the last
    peak matched by the previous fragment (or where the previous scan stopped), which only matters where the windows
    of consecutive fragments overlap.

    :param fragments: fragments sorted by m/z (fragment_arrays)
    :param peaks_masses: experimental m/z in ascending order, precursor window removed
    :param tmt_n_term: 2 if there is a tmt on the n-terminus
    :param unmod_sequence: unmodified peptide sequence
    :param featured_ions: ion types, fragments["ion"] indexes them
    :return: fragment and peak index of every matched ion without neutral loss (in the order of match_peaks), number
        of matched neutral loss peaks
    """
    lo = np.searchsorted(peaks_masses, fragments["min_mass"], side="left")
    hi = np.searchsorted(peaks_masses, fragments["max_mass"], side="right")
    start = lo.copy()
    matched = lo < hi
    previous_hi = np.concatenate(([0], hi[:-1]))
    for k in np.flatnonzero(lo < previous_hi):
        start[k] = max(previous_hi[k] - int(matched[k - 1]), lo[k])
        matched[k] = start[k] < hi[k]

    counts = np.where(matched, hi - start, 0)
    fragment_idx = np.repeat(np.arange(len(counts)), counts)
    peak_idx = np.arange(len(fragment_idx)) - np.repeat(np.cumsum(counts) - counts, counts) + start[fragment_idx]

    # unstable b1 (sequence not starting with R, H, K, S) and a2 (Q, N or C in the first two residues) are not counted
    unstable = np.zeros(len(counts), dtype=bool)
    if tmt_n_term != 2:
        if "b" in featured_ions and not unmod_sequence.startswith(("R", "H", "K", "S")):
            unstable |= (fragments["ion"] == featured_ions.index("b")) & (fragments["no"] == 1)
        if "a" in featured_ions and any(aa in unmod_sequence[:2] for aa in "QNC"):
            unstable |= (fragments["ion"] == featured_ions.index("a")) & (fragments["no"] == 2)
    keep = ~unstable[fragment_idx]
    neutral_loss = fragments["neutral_loss"][fragment_idx]
    count_annotated_nl = int(np.count_nonzero(keep & neutral_loss))
    keep &= ~neutral_loss
    return fragment_idx[keep], peak_idx[keep], count_annotated_nl


def _annotate_linear_spectrum_arrays(
    spectrum: np.ndarray,
    index_columns: dict[str, int],
    mod_seq_column: str,
    mass_tolerance: float | None,
    unit_mass_tolerance: str | None,
    custom_mods: dict[str, float] | None,
    featured_ions: list[str],
    multifrag: bool,
    p_window: float,
    add_neutral_losses: bool,
    phospho_loss_residues: str,
    plain_loss_residues: str | None,
):
    """
    Annotate a linear peptide spectrum, array version of _annotate_linear_spectrum.

    Same result as initialize_peaks, match_peaks, handle_multiple_matches (closest peak per fragment) and
    generate_annotation_matrix, without one dictionary or DataFrame per spectrum. Needs peaks in ascending m/z.

    :param spectrum: Spectrum to be annotated
    :param index_columns: Index columns of the spectrum
    :param mod_seq_column: column of the modified sequence
    :param mass_tolerance: Mass tolerance for calculating min and max mass
    :param unit_mass_tolerance: Unit for the mass tolerance (da or ppm)
    :param custom_mods: mapping of custom UNIMOD string identifiers ('[UNIMOD:xyz]') to their mass
    :param featured_ions: list of ions to be annotated
    :param multifrag: flag to indicate whether to annotate multifrag or not
    :param p_window: peak exclusion window around the precursor (da)
    :param add_neutral_losses: flag to indicate whether to annotate neutral losses or not
    :param phospho_loss_residues: phosphorylated residues annotated with the loss of phosphoric acid
    :param plain_loss_residues: if given, the number of annotated and expected neutral losses with these residues
        instead of phospho_loss_residues are added to the result
    :return: Annotated spectrum
    """
    sequence = spectrum[index_columns[mod_seq_column]]
    precursor_charge = spectrum[index_columns["PRECURSOR_CHARGE"]]
    peak_args = {
        "mass_analyzer": spectrum[index_columns["MASS_ANALYZER"]],
        "charge": precursor_charge,
        "mass_tolerance": mass_tolerance,
        "unit_mass_tolerance": unit_mass_tolerance,
        "featured_ions": featured_ions,
        "multifrag": multifrag,
        "p_window": p_window,
        "custom_mods": custom_mods,
        "add_neutral_losses": add_neutral_losses,
    }
    fragments, tmt_n_term, unmod_sequence, calc_mass, expected_nl, p_window_bounds = fragment_arrays(
        sequence, phospho_loss_residues=phospho_loss_residues, **peak_args
    )
    peaks_masses = np.asarray(spectrum[index_columns["MZ"]])
    peaks_intensity = np.asarray(spectrum[index_columns["INTENSITIES"]])
    outside_window = ~((peaks_masses > p_window_bounds[0]) & (peaks_masses < p_window_bounds[1]))
    if not outside_window.all():
        peaks_masses, peaks_intensity = peaks_masses[outside_window], peaks_intensity[outside_window]
    fragment_idx, peak_idx, count_annotated_nl = _match_fragments(
        fragments, peaks_masses, tmt_n_term, unmod_sequence, featured_ions
    )

    plain_counts: tuple[int, ...] = ()
    if plain_loss_residues is not None:
        plain_counts = (count_annotated_nl, expected_nl)
        extra_residues = "".join(set(phospho_loss_residues) - set(plain_loss_residues))
        if add_neutral_losses and extra_residues and re.search(rf"[{extra_residues}]\[UNIMOD:21\]", sequence):
            plain_fragments, _, _, _, plain_expected_nl, _ = fragment_arrays(
                sequence, phospho_loss_residues=plain_loss_residues, **peak_args
            )
            _, _, plain_annotated_nl = _match_fragments(
                plain_fragments, peaks_masses, tmt_n_term, unmod_sequence, featured_ions
            )
            plain_counts = (plain_annotated_nl, plain_expected_nl)

    charge_const = 3
    n_ion_types = len(featured_ions)
    if len(fragment_idx) == 0:
        vec_length = len(constants.ION_DIC) if multifrag else (constants.SEQ_LEN - 1) * charge_const * n_ion_types
        return (np.full(vec_length, 0.0), np.full(vec_length, 0.0), calc_mass, 0, 0, 0, np.nan) + (
            (0, 0) if plain_counts else ()
        )

    intensity = peaks_intensity[peak_idx].astype(float)
    max_intensity = max(0.0, float(intensity.max()))
    exp_mass = peaks_masses[peak_idx]
    # several peaks matched to the same fragment: keep the one closest to the theoretical m/z
    mass_diff = np.abs(exp_mass - fragments["mass"][fragment_idx])
    order = np.lexsort((mass_diff, fragment_idx))
    first = np.concatenate(([True], fragment_idx[order][1:] != fragment_idx[order][:-1]))
    best = order[first]
    removed_peaks = len(fragment_idx) - len(best)

    fragment_best = fragment_idx[best]
    if multifrag:
        position = fragments["ion_row"][fragment_best]
        charge = int(precursor_charge)
    else:
        ion_position = np.array([featured_ions.index(ion.split("-", 1)[0]) for ion in featured_ions])
        position = (
            (fragments["no"][fragment_best] - 1) * charge_const * n_ion_types
            + (fragments["charge"][fragment_best] - 1)
            + charge_const * ion_position[fragments["ion"][fragment_best]]
        )
        charge = int(min(precursor_charge, 3))
    intensities = _empty_annotation(len(unmod_sequence), charge, tuple(featured_ions), multifrag).copy()
    mass = intensities.copy()
    inside = position < len(intensities)
    intensities[position[inside]] = intensity[best][inside] / max_intensity
    mass[position[inside]] = exp_mass[best][inside]

    return (
        intensities,
        mass,
        calc_mass,
        removed_peaks,
        count_annotated_nl,
        expected_nl,
        max_intensity,
    ) + plain_counts


def _annotate_crosslinked_spectrum(
    spectrum: np.ndarray,
    index_columns: dict[str, int],
    crosslinker_type: str,
    mass_tolerance: float | None = None,
    unit_mass_tolerance: str | None = None,
    custom_mods: dict[str, float] | None = None,
):
    """
    Annotate a crosslinked peptide spectrum.

    :param spectrum: Spectrum to be annotated
    :param index_columns: Index columns of the spectrum
    :param crosslinker_type: Type of crosslinker used
    :param mass_tolerance: Mass tolerance for calculating min and max mass
    :param unit_mass_tolerance: Unit for the mass tolerance (da or ppm)
    :param custom_mods: mapping of custom UNIMOD string identifiers ('[UNIMOD:xyz]') to their mass
    :raises ValueError: if unsupported crosslinker type was supplied.

    :return: Annotated spectrum
    """
    if crosslinker_type in ["BS3", "DSS"]:
        non_cl_xl = True
    elif crosslinker_type in ["DSSO", "DSBU", "BUURBU"]:
        non_cl_xl = False
    else:
        raise ValueError(f"Unsupported crosslinker type provided: {crosslinker_type}")

    def _xl_annotation_workflow(seq_id: str, non_cl_xl: bool):
        inputs = [
            spectrum[index_columns[f"MODIFIED_SEQUENCE_{seq_id[0]}"]],
            spectrum[index_columns["MASS_ANALYZER"]],
            spectrum[index_columns[f"CROSSLINKER_POSITION_{seq_id[0]}"]],
            crosslinker_type,
            mass_tolerance,
            unit_mass_tolerance,
        ]
        if non_cl_xl:
            inputs.append(spectrum[index_columns[f"MODIFIED_SEQUENCE_{seq_id[1]}"]])
            array_size = 174

        else:
            array_size = 348
        inputs.append(custom_mods)
        fragments_meta_data, tmt_n_term, unmod_sequence, calc_mass = initialize_peaks_xl(*inputs)
        matched_peaks, annotated_nl, _ = match_peaks(
            fragments_meta_data,
            np.array(spectrum[index_columns["INTENSITIES"]]),
            np.array(spectrum[index_columns["MZ"]]),  # Convert to numpy array
            tmt_n_term,
            unmod_sequence,
        )

        if len(matched_peaks) == 0:
            intensities = np.full(array_size, 0.0)
            mass = np.full(array_size, 0.0)
            removed_peaks = 0
        else:
            matched_peaks, removed_peaks = handle_multiple_matches(matched_peaks)
            if non_cl_xl:
                intensities, mass = generate_annotation_matrix(
                    matched_peaks, unmod_sequence, spectrum[index_columns["PRECURSOR_CHARGE"]]
                )
            else:
                intensities, mass = generate_annotation_matrix_xl(
                    matched_peaks, unmod_sequence, spectrum[index_columns[f"CROSSLINKER_POSITION_{seq_id[0]}"]]
                )

        return intensities, mass, removed_peaks, calc_mass

    intensities_a, mass_a, removed_peaks_a, calc_mass_a = _xl_annotation_workflow(seq_id="AB", non_cl_xl=non_cl_xl)
    intensities_b, mass_b, removed_peaks_b, calc_mass_b = _xl_annotation_workflow(seq_id="BA", non_cl_xl=non_cl_xl)

    return intensities_a, intensities_b, mass_a, mass_b, calc_mass_a, calc_mass_b, removed_peaks_a, removed_peaks_b
    intensities_a, mass_a, removed_peaks_a, calc_mass_a = _xl_annotation_workflow(seq_id="AB", non_cl_xl=non_cl_xl)
    intensities_b, mass_b, removed_peaks_b, calc_mass_b = _xl_annotation_workflow(seq_id="BA", non_cl_xl=non_cl_xl)

    return intensities_a, intensities_b, mass_a, mass_b, calc_mass_a, calc_mass_b, removed_peaks_a, removed_peaks_b
