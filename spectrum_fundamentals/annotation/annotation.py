import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from spectrum_fundamentals import constants
from spectrum_fundamentals.fragments import initialize_peaks

logger = logging.getLogger(__name__)


def match_peaks(
    fragments_meta_data: List[dict],
    peaks_intensity: np.ndarray,
    peaks_masses: np.ndarray,
    tmt_n_term: int,
    unmod_sequence: str,
    charge: int,
) -> List[Dict[str, Union[str, int, float]]]:
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
    max_intensity = 1.0
    row_list = []
    temp_list = []
    next_start_peak = 0
    seq_len = len(unmod_sequence)
    matched_peak = False
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
                not (fragment["ion_type"] == "b" and fragment_no == 1)
                or (unmod_sequence[0] == "R" or unmod_sequence[0] == "H" or unmod_sequence[0] == "K")
                and (tmt_n_term == 1)
            ):
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
                if peak_intensity > max_intensity and fragment_no < seq_len:
                    max_intensity = float(peak_intensity)
            matched_peak = True
            next_start_peak = start_peak
            start_peak += 1
    for row in row_list:
        row["intensity"] = float(row["intensity"]) / max_intensity
        temp_list.append(row)
    return temp_list


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
    un_annot_spectra: pd.DataFrame, mass_tolerance: Optional[float] = None, unit_mass_tolerance: Optional[str] = None
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
    :return: a Pandas DataFrame containing the annotated spectra with meta data
    """
    raw_file_annotations = []
    index_columns = {col: un_annot_spectra.columns.get_loc(col) for col in un_annot_spectra.columns}
    for row in un_annot_spectra.values:
        results = parallel_annotate(row, index_columns, mass_tolerance, unit_mass_tolerance)
        if not results:
            continue
        raw_file_annotations.append(results)
    results_df = pd.DataFrame(raw_file_annotations)
    results_df.columns = ["INTENSITIES", "MZ", "CALCULATED_MASS", "removed_peaks"]
    logger.info(f"Removed {results_df['removed_peaks'].describe()} redundant peaks")

    return results_df


def generate_annotation_matrix(
    matched_peaks: pd.DataFrame, unmod_seq: str, charge: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the annotation matrix in the prosit format from matched peaks.

    :param matched_peaks: matched peaks needed to be converted
    :param unmod_seq: Un modified peptide sequence
    :param charge: Precursor charge
    :return: numpy array of intensities and numpy array of masses
    """
    intensity = np.full(constants.VEC_LENGTH, -1.0)
    mass = np.full(constants.VEC_LENGTH, -1.0)

    # change values to zeros
    if len(unmod_seq) < constants.SEQ_LEN:
        peaks_range = range(0, ((len(unmod_seq) - 1) * 6))
    else:
        peaks_range = range(0, ((constants.SEQ_LEN - 1) * 6))

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
        if peak[ion_type] == "y":
            peak_pos = ((peak[no_col] - 1) * 6) + (peak[charge_col] - 1)
        else:
            peak_pos = ((peak[no_col] - 1) * 6) + (peak[charge_col] - 1) + 3

        if peak_pos >= constants.VEC_LENGTH:
            continue
        intensity[peak_pos] = peak[intensity_col]
        mass[peak_pos] = peak[exp_mass_col]

    if len(unmod_seq) < constants.SEQ_LEN:
        mask_peaks = range((len(unmod_seq) - 1) * 6, ((len(unmod_seq) - 1) * 6) + 6)
        intensity[mask_peaks] = -1.0
        mass[mask_peaks] = -1.0

    return intensity, mass


def parallel_annotate(
    spectrum: np.ndarray,
    index_columns: dict,
    mass_tolerance: Optional[float] = None,
    unit_mass_tolerance: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, int]]:
    """
    Perform parallel annotation of a spectrum.

    This function takes a spectrum and its index columns and performs parallel annotation of the spectrum. It starts by
    initializing the peaks and extracting necessary data from the spectrum. It then matches the peaks to the spectrum and
    generates an annotation matrix based on the matched peaks. If there are multiple matches found, it removes the redundant
    matches. Finally, it returns annotated spectrum with meta data including intensity values, masses, calculated masses,
    and any peaks that were removed. The function is designed to run in different threads to speed up the annotation pipeline.

    :param spectrum: a np.ndarray that contains the spectrum to be annotated
    :param index_columns: a dictionary that contains the index columns of the spectrum
    :param mass_tolerance: mass tolerance to calculate min and max mass
    :param unit_mass_tolerance: unit for the mass tolerance (da or ppm)
    :return: a tuple containing intensity values (np.ndarray), masses (np.ndarray), calculated mass (float),
             and any removed peaks (List[str])
    """
    mod_seq_column = "MODIFIED_SEQUENCE"
    if "MODIFIED_SEQUENCE_MSA" in index_columns:
        mod_seq_column = "MODIFIED_SEQUENCE_MSA"

    fragments_meta_data, tmt_n_term, unmod_sequence, calc_mass = initialize_peaks(
        spectrum[index_columns[mod_seq_column]],
        spectrum[index_columns["MASS_ANALYZER"]],
        spectrum[index_columns["PRECURSOR_CHARGE"]],
        mass_tolerance,
        unit_mass_tolerance,
    )
    if not unmod_sequence:
        return None
    matched_peaks = match_peaks(
        fragments_meta_data,
        spectrum[index_columns["INTENSITIES"]],
        spectrum[index_columns["MZ"]],
        tmt_n_term,
        unmod_sequence,
        spectrum[index_columns["PRECURSOR_CHARGE"]],
    )
    if len(matched_peaks) == 0:
        intensity = np.full(174, 0.0)
        mass = np.full(174, 0.0)
        return intensity, mass, calc_mass, 0
    matched_peaks, removed_peaks = handle_multiple_matches(matched_peaks)
    intensities, mass = generate_annotation_matrix(
        matched_peaks, unmod_sequence, spectrum[index_columns["PRECURSOR_CHARGE"]]
    )
    return intensities, mass, calc_mass, removed_peaks
