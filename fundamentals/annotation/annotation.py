from operator import itemgetter

import pandas as pd
from joblib import Parallel, delayed
import numpy as np

from fundamentals import constants
from fundamentals.fragments import initialize_peaks, get_modifications

MIN_CHARGE = 1
NUM_CORES = 5

#TODO Check if this function is used or can be deleted
def maxquant_parser(p):
    if p[0] == "(":
        p = p[4] + p[:4] + p[5:]
    n = len(p)
    i = 0
    while i < n:
        if i < n - 3 and p[i + 1] == "(":
            j = p[i + 2:].index(")")
            offset = i + j + 3
            yield p[i:offset]
            i = offset
        else:
            yield p[i]
            i += 1

#TODO Check if this function is used or can be deleted
def str_to_integer(sequences, parser=maxquant_parser):
    """

    :param sequences:
    :param parser:
    :return:
    """
    array = np.zeros([len(sequences), 30], dtype=int)
    for i, sequence in enumerate(sequences):
        for j, symbol in enumerate(parser(sequence)):
            # if symbol == "(":
            # print(i, j, sequence)
            try:
                array[i, j] = constants.ALPHABET[symbol]
            except Exception:
                print(sequence)
    return array


def match_peaks(fragments_meta_data: list, peaks_intensity: np,
                peaks_masses: np, tmt_n_term: int, unmod_sequence: str, charge: int):
    """
    Matching experimental peaks with theoretical fragment ions
    :param fragments_meta_data: Fragments ions meta data eg. ion type, number, theo_mass...
    :param peaks_intensity: Experimental peaks intensities.
    :param peaks_masses: Experimental peaks masses.
    :param tmt_n_term: Flag to check if there is tmt modification on n_terminus 1: no_tmt, 2:tmt
    :param unmod_sequence: Unmodified peptide sequence.
    :param charge: Precursor charge.
    :return: List of matched/annotated peaks
    """
    start_peak = 0
    no_of_peaks = len(peaks_intensity)
    max_intensity = 1
    row_list = []
    temp_list = []
    for fragment in fragments_meta_data:
        min_mass = fragment['min_mass']
        max_mass = fragment['max_mass']
        fragment_no = fragment['no']
        while start_peak < no_of_peaks:
            peak_mass = peaks_masses[start_peak]
            peak_intensity = peaks_intensity[start_peak]

            if peak_mass > max_mass:
                break
            if peak_mass < min_mass:
                start_peak+=1
            if min_mass <= peak_mass <= max_mass:
                if fragment['ion_type'] == 'b' and fragment_no == 1:
                    if (unmod_sequence[0] == 'R' or unmod_sequence[0] == 'H' or unmod_sequence[0] == 'K') and (
                            tmt_n_term == 1):
                        row_list.append(
                            {'ion_type': fragment['ion_type'], 'no': fragment_no, 'charge': charge,
                             'exp_mass': peak_mass, 'theoretical_mass': fragment['mass'], 'intensity': peak_intensity})
                        if peak_intensity > max_intensity:
                            max_intensity = float(peak_intensity)
                else:
                    row_list.append(
                        {'ion_type': fragment['ion_type'], 'no': fragment_no, 'charge': charge,
                         'exp_mass': peak_mass, 'theoretical_mass': fragment['mass'], 'intensity': peak_intensity})
                    if peak_intensity > max_intensity:
                        max_intensity = float(peak_intensity)
                start_peak+=1
    for row in row_list:
        row['intensity'] = float(row['intensity']) / max_intensity
        temp_list.append(row)
    return temp_list


def handle_multiple_matches(matched_peaks: list, sort_by: str = 'intensity'):
    """
    Here we handle if multiple peaks were matched to the same fragment ion.
    We will resolve this based on the sort_by parameter.
    :param matched_peaks: all matched peaks we have for the spectrum.
    :param sort_by: choose how to sort peaks e.g. intensity, mass_diff
    """
    matched_peaks_df = pd.DataFrame(matched_peaks)
    if sort_by == 'mass_diff':
        matched_peaks_df['mass_diff'] = abs(matched_peaks_df['exp_mass'] - matched_peaks_df['theoretical_mass'])
        matched_peaks_df = matched_peaks_df.sort_values(by='mass_diff', ascending=True)
    else:
        matched_peaks_df = matched_peaks_df.sort_values(by='intensity', ascending=False)

    matched_peaks_df = matched_peaks_df.drop_duplicates(subset=['ion_type', 'no'], keep="first")
    return matched_peaks_df


def annotate_spectra(un_annot_spectra: pd.DataFrame):
    """
    The base method for annotating spectra.
    :param un_annot_spectra: dataframe of raw peaks and metadata.
    :return: List of annotated spectra.
    """
    raw_file_annotations = []
    # modified_sequence_column = un_annot_spectra.columns.get_loc('MODIFIED_SEQUENCE')
    # mass_analyzer_column = un_annot_spectra.columns.get_loc('MASS_ANALYZER')
    # charge_column = un_annot_spectra.columns.get_loc('PRECURSOR_CHARGE')
    # peaks_intensities_column = un_annot_spectra.columns.get_loc('INTENSITIES')
    # peaks_mz_column = un_annot_spectra.columns.get_loc('MZ')
    #
    # index_columns = {'mod_sequence'}

    index_columns = {col: un_annot_spectra.columns.get_loc(col) for col in un_annot_spectra.columns}
    print(index_columns)
    for row in un_annot_spectra.values:
            results = parallel_annotate(row, index_columns)
            raw_file_annotations.append(results)
    results_df= pd.DataFrame()
    results_df = results_df.append(raw_file_annotations)
    results_df.columns = ["INTENSITIES", "MZ"]

    return results_df


def generate_annotation_matrix(matched_peaks, unmod_seq: str, charge: int):
    """
    Generate the annotation matrix in the prosit format from matched peaks.
    :param matched_peaks: matched peaks needed to be converted.
    :param unmod_seq: Un modified peptide sequence
    :param charge: Precursor charge
    :return: numpy array of intensities and  numpy array of masses
    """
    intensity = np.full(174, -1.0)
    mass = np.full(174, -1.0)

    # change values to zeros
    if len(unmod_seq) < 30:
        peaks_range = range(0, (len(unmod_seq) - 1) * 6)
    else:
        peaks_range = range(0, (len(unmod_seq) - 1) * 6)
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
        if peak[ion_type] == 'y':
            peak_pos = (peak[no_col] - 1) * 6 + (peak[charge_col] - 1)
        else:
            peak_pos = (peak[no_col] - 1) * 6 + (peak[charge_col] - 1) + 3

        if peak_pos >= 174:
            break
        intensity[peak_pos] = peak[intensity_col]
        mass[peak_pos] = peak[exp_mass_col]

    if len(unmod_seq) < 30:
        mask_peaks = range((len(unmod_seq) - 1) * 6, ((len(unmod_seq) - 1) * 6) + 6)
        intensity[mask_peaks] = -1.0
        mass[mask_peaks] = -1.0

    return intensity, mass


def parallel_annotate(spectrum, index_columns):
    """
    Parallelize the annotation pipeline, here it should annotate spectra in different threads.
    :param spectrum: spectrum to be annotated.
    :return: annotated spectrum with meta data.
    """
    fragments_meta_data, tmt_n_term, unmod_sequence = initialize_peaks(spectrum[index_columns['MODIFIED_SEQUENCE']], spectrum[index_columns['MASS_ANALYZER']],
                                                                       spectrum[index_columns['PRECURSOR_CHARGE']])
    matched_peaks = match_peaks(fragments_meta_data, spectrum[index_columns['INTENSITIES']], spectrum[index_columns['MZ']], tmt_n_term, unmod_sequence,
                                spectrum[index_columns['PRECURSOR_CHARGE']])
    if len(matched_peaks) == 0:
        intensity = np.full(174, 0.0)
        mass = np.full(174, 0.0)
        return intensity, mass
    matched_peaks = handle_multiple_matches(matched_peaks)
    intensities, mass = generate_annotation_matrix(matched_peaks, unmod_sequence, spectrum[index_columns['PRECURSOR_CHARGE']])
    return intensities,mass
