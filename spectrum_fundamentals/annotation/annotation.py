import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from spectrum_fundamentals import constants
from spectrum_fundamentals.fragments import initialize_peaks, initialize_peaks_xl, retrieve_ion_types

logger = logging.getLogger(__name__)


def match_peaks(
    fragments_meta_data: List[dict],
    peaks_intensity: np.ndarray,
    peaks_masses: np.ndarray,
    tmt_n_term: int,
    unmod_sequence: str,
    charge: int,
) -> Tuple[List[Dict[str, Union[str, int, float]]], int]:
    """
    Matching experimental peaks with theoretical fragment ions.

    :param fragments_meta_data: Fragments ions meta data eg. ion type, number, theo_mass...
    :param peaks_intensity: Experimental peaks intensities
    :param peaks_masses: Experimental peaks masses
    :param tmt_n_term: Flag to check if there is tmt modification on n_terminus 1: no_tmt, 2:tmt
    :param unmod_sequence: Unmodified peptide sequence
    :param charge: Precursor charge
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

            if peak_mass > max_mass:
                break
            if peak_mass < min_mass:
                start_peak += 1
                continue
            if (
                not (fragment["ion_type"][0] == "b" and fragment_no == 1)
                or (unmod_sequence[0] == "R" or unmod_sequence[0] == "H" or unmod_sequence[0] == "K")
                or (tmt_n_term == 2)
            ):
                # For now only counting neutral loss peaks this can change with different models later
                if fragment["neutral_loss"] == "":
                    row_list.append(
                        {
                            "ion_type": fragment["ion_type"],
                            "no": fragment_no,
                            "charge": fragment["charge"],
                            "exp_mass": peak_mass,
                            "theoretical_mass": fragment["mass"],
                            "intensity": peak_intensity,
                        }
                    )
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
    return temp_list, count_annotated_nl


def handle_multiple_matches(
    matched_peaks: List[Dict[str, Union[str, int, float]]], sort_by: str = "mass_diff"
) -> Tuple[pd.DataFrame, int]:
    """
    Resolve cases where multiple peaks have been matched to the same fragment ion.

    This function takes a list of dictionaries representing matched peaks and resolves cases where multiple peaks have
    been matched to the same fragment ion. The function sorts the peaks based on the provided `sort_by` parameter and
    removes duplicate matches based on ion type, ion number, and charge state.

    :param matched_peaks: A list of dictionaries, each representing a matched peak. Each dictionary must contain the
                          following keys: 'ion_type', 'no', 'charge', 'exp_mass', 'theoretical_mass', and 'intensity'.
    :param sort_by: A string indicating the criterion to use when sorting matched peaks. Valid options are:
                    'mass_diff' (sort by absolute difference between experimental and theoretical mass, ascending order),
                    'intensity' (sort by intensity, descending order), and 'exp_mass' (sort by experimental mass,
                    descending order).
    :raises ValueError: If an unsupported value is passed to `sort_by`.
    :return: A tuple containing a DataFrame of matched peaks (with duplicates removed) and an integer indicating the
             number of duplicate matches that were removed.
    """
    matched_peaks_df = pd.DataFrame(matched_peaks)
    if sort_by == "mass_diff":
        matched_peaks_df["mass_diff"] = abs(matched_peaks_df["exp_mass"] - matched_peaks_df["theoretical_mass"])
        matched_peaks_df = matched_peaks_df.sort_values(by="mass_diff", ascending=True)
    elif sort_by == "intensity":
        matched_peaks_df = matched_peaks_df.sort_values(by="intensity", ascending=False)
    elif sort_by == "exp_mass":
        matched_peaks_df = matched_peaks_df.sort_values(by="exp_mass", ascending=False)
    else:
        raise ValueError(f"Unsupported value for sort_by supplied: {sort_by}")

    original_length = len(matched_peaks_df.index)
    matched_peaks_df = matched_peaks_df.drop_duplicates(subset=["ion_type", "no", "charge"], keep="first")
    # matched_peaks_df = matched_peaks_df[matched_peaks_df['intensity']>0.01]
    length_after_matches = len(matched_peaks_df.index)
    return matched_peaks_df, (original_length - length_after_matches)


def annotate_spectra(
    un_annot_spectra: pd.DataFrame,
    mass_tolerance: Optional[float] = None,
    unit_mass_tolerance: Optional[str] = None,
    custom_mods: Optional[Dict[str, float]] = None,
    fragmentation_method: str = "HCD",
    annotate_neutral_loss: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Annotate a set of spectra.

    This function takes a DataFrame of raw peaks and metadata, and for each spectrum, it calls the `parallel_annotate` function
    to annotate the spectrum and extract the necessary information. If there are any redundant peaks found in the annotation
    process, the function removes them and logs the information. Finally, it returns a Pandas DataFrame containing the annotated
    spectra with meta data.

    The returned DataFrame has the following columns:
    - INTENSITIES: a NumPy array containing the intensity values of each peak in the annotated spectrum
    - MZ: a NumPy array containing the m/z values of each peak in the annotated spectrum
    - CALCULATED_MASS: a float representing the calculated mass of the spectrum
    - removed_peaks: a NumPy array containing the indices of any peaks that were removed during the annotation process

    :param un_annot_spectra: a Pandas DataFrame containing the raw peaks and metadata to be annotated
    :param mass_tolerance: mass tolerance to calculate min and max mass
    :param unit_mass_tolerance: unit for the mass tolerance (da or ppm)
    :param fragmentation_method: fragmentation method that was used
    :param custom_mods: mapping of custom UNIMOD string identifiers ('[UNIMOD:xyz]') to their mass
    :param annotate_neutral_loss: flag to indicate whether to annotate neutral losses or not
    :return: a Pandas DataFrame containing the annotated spectra with meta data
    """
    raw_file_annotations = []
    index_columns = {col: un_annot_spectra.columns.get_loc(col) for col in un_annot_spectra.columns}
    for row in un_annot_spectra.values:
        results = parallel_annotate(
            row,
            index_columns,
            mass_tolerance,
            unit_mass_tolerance,
            fragmentation_method=fragmentation_method,
            custom_mods=custom_mods,
            annotate_neutral_losses=annotate_neutral_loss,
        )
        if not results:
            continue
        raw_file_annotations.append(results)
    results_df = pd.DataFrame(raw_file_annotations)

    if "CROSSLINKER_TYPE" not in index_columns:
        results_df.columns = [
            "INTENSITIES",
            "MZ",
            "CALCULATED_MASS",
            "removed_peaks",
            "ANNOTATED_NL_COUNT",
            "EXPECTED_NL_COUNT",
        ]
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
    Determines the positions of all potential normal and xl fragments within the vector generated by generate_annotation_matrix.

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
) -> Tuple[np.ndarray, np.ndarray]:
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


def generate_annotation_matrix(
    matched_peaks: pd.DataFrame, unmod_seq: str, charge: int, fragmentation_method: str = "HCD"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the annotation matrix in the prosit format from matched peaks.

    :param matched_peaks: matched peaks needed to be converted
    :param unmod_seq: Un modified peptide sequence
    :param charge: Precursor charge
    :param fragmentation_method: fragmentation method that was used
    :return: numpy array of intensities and numpy array of masses
    """
    ion_types = retrieve_ion_types(fragmentation_method)
    charge_const = 3
    vec_length = (constants.SEQ_LEN - 1) * charge_const * len(ion_types)

    intensity = np.full(vec_length, -1.0)
    mass = np.full(vec_length, -1.0)

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

    intensity[available_peaks] = 0.0
    mass[available_peaks] = 0.0

    ion_type = matched_peaks.columns.get_loc("ion_type")
    no_col = matched_peaks.columns.get_loc("no")
    charge_col = matched_peaks.columns.get_loc("charge")
    intensity_col = matched_peaks.columns.get_loc("intensity")
    exp_mass_col = matched_peaks.columns.get_loc("exp_mass")

    for peak in matched_peaks.values:
        ion_type_index = ion_types.index(peak[ion_type].split("-", 1)[0])
        peak_pos = ((peak[no_col] - 1) * charge_const * len(ion_types)) + (peak[charge_col] - 1) + 3 * ion_type_index

        if peak_pos >= constants.VEC_LENGTH:
            continue
        intensity[peak_pos] = peak[intensity_col]
        mass[peak_pos] = peak[exp_mass_col]

    if len(unmod_seq) < constants.SEQ_LEN:
        mask_peaks = range(
            (len(unmod_seq) - 1) * charge_const * len(ion_types), ((len(unmod_seq)) * charge_const * len(ion_types))
        )
        intensity[mask_peaks] = -1.0
        mass[mask_peaks] = -1.0

    return intensity, mass


def parallel_annotate(
    spectrum: np.ndarray,
    index_columns: Dict[str, int],
    mass_tolerance: Optional[float] = None,
    unit_mass_tolerance: Optional[str] = None,
    custom_mods: Optional[Dict[str, float]] = None,
    fragmentation_method: str = "HCD",
    annotate_neutral_losses: Optional[bool] = False,
) -> Optional[
    Union[
        Tuple[np.ndarray, np.ndarray, float, int, int, int],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, int, int],
    ]
]:
    """
    Perform parallel annotation of a spectrum.

    This function takes a spectrum and its index columns and performs parallel annotation of the spectrum.
    It starts by initializing the peaks and extracting necessary data from the spectrum.
    It then matches the peaks to the spectrum and generates an annotation matrix based on the matched peaks.
    If there are multiple matches found, it removes the redundant matches.
    Finally, it returns annotated spectrum with meta data including intensity values, masses, calculated masses,
    and any peaks that were removed. The function is designed to run in different threads to speed up the annotation pipeline.

    :param spectrum: a np.ndarray that contains the spectrum to be annotated
    :param index_columns: a dictionary that contains the index columns of the spectrum
    :param mass_tolerance: mass tolerance to calculate min and max mass
    :param unit_mass_tolerance: unit for the mass tolerance (da or ppm)
    :param custom_mods: mapping of custom UNIMOD string identifiers ('[UNIMOD:xyz]') to their mass
    :param fragmentation_method: fragmentation method that was used
    :param annotate_neutral_losses: flag to indicate whether to annotate neutral losses or not
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
            custom_mods=custom_mods,
            add_neutral_losses=annotate_neutral_losses,
        )

    if (spectrum[index_columns["PEPTIDE_LENGTH_A"]] > 30) or (spectrum[index_columns["PEPTIDE_LENGTH_B"]] > 30):
        return None
    return _annotate_crosslinked_spectrum(
        spectrum, index_columns, spectrum[xl_type_col], mass_tolerance, unit_mass_tolerance, custom_mods=custom_mods
    )


def _annotate_linear_spectrum(
    spectrum: np.ndarray,
    index_columns: Dict[str, int],
    mass_tolerance: Optional[float],
    unit_mass_tolerance: Optional[str],
    custom_mods: Optional[Dict[str, float]] = None,
    fragmentation_method: str = "HCD",
    add_neutral_losses: Optional[bool] = False,
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
    :return: Annotated spectrum
    """
    mod_seq_column = "MODIFIED_SEQUENCE"
    if "MODIFIED_SEQUENCE_MSA" in index_columns:
        mod_seq_column = "MODIFIED_SEQUENCE_MSA"

    fragments_meta_data, tmt_n_term, unmod_sequence, calc_mass, expected_nl = initialize_peaks(
        sequence=spectrum[index_columns[mod_seq_column]],
        mass_analyzer=spectrum[index_columns["MASS_ANALYZER"]],
        charge=spectrum[index_columns["PRECURSOR_CHARGE"]],
        mass_tolerance=mass_tolerance,
        unit_mass_tolerance=unit_mass_tolerance,
        fragmentation_method=fragmentation_method,
        custom_mods=custom_mods,
        add_neutral_losses=add_neutral_losses,
    )
    matched_peaks, count_annotated_nl = match_peaks(
        fragments_meta_data,
        spectrum[index_columns["INTENSITIES"]],
        spectrum[index_columns["MZ"]],
        tmt_n_term,
        unmod_sequence,
        spectrum[index_columns["PRECURSOR_CHARGE"]],
    )

    ion_types = retrieve_ion_types(fragmentation_method)
    charge_const = 3
    vec_length = (constants.SEQ_LEN - 1) * charge_const * len(ion_types)

    if len(matched_peaks) == 0:
        intensity = np.full(vec_length, 0.0)
        mass = np.full(vec_length, 0.0)
        return intensity, mass, calc_mass, 0, 0, 0

    matched_peaks, removed_peaks = handle_multiple_matches(matched_peaks)
    intensities, mass = generate_annotation_matrix(
        matched_peaks, unmod_sequence, spectrum[index_columns["PRECURSOR_CHARGE"]], fragmentation_method
    )
    return intensities, mass, calc_mass, removed_peaks, count_annotated_nl, expected_nl


def _annotate_crosslinked_spectrum(
    spectrum: np.ndarray,
    index_columns: Dict[str, int],
    crosslinker_type: str,
    mass_tolerance: Optional[float] = None,
    unit_mass_tolerance: Optional[str] = None,
    custom_mods: Optional[Dict[str, float]] = None,
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
        matched_peaks, annotated_nl = match_peaks(
            fragments_meta_data,
            np.array(spectrum[index_columns["INTENSITIES"]]),
            np.array(spectrum[index_columns["MZ"]]),  # Convert to numpy array
            tmt_n_term,
            unmod_sequence,
            spectrum[index_columns["PRECURSOR_CHARGE"]],
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
