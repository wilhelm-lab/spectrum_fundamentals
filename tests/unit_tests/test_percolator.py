import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse

import spectrum_fundamentals.constants as constants
import spectrum_fundamentals.metrics.percolator as perc


class TestFdrs:
    """Class to test FDRs."""

    def test_calculate_fdrs(self):
        """Test calculcate_fdrs."""
        t = perc.TargetDecoyLabel.TARGET
        d = perc.TargetDecoyLabel.DECOY
        sorted_labels = np.array([t, t, d, d])
        np.testing.assert_almost_equal(
            perc.Percolator.calculate_fdrs(sorted_labels), [0.3333333, 0.3333333, 0.66666667, 1.0]
        )

    def test_get_indices_below_fdr_none(self):
        """Test get_indices_below_fdr."""
        percolator = perc.Percolator(metadata=pd.DataFrame(), input_type="rescore")
        percolator.metrics_val["Score"] = [0, 3, 2, 1]
        percolator.target_decoy_labels = [-1, -1, 1, -1]
        """
        idx Score  Label  fdr
        1      3     -1  2.0
        2      2      1  1.0
        3      1     -1  1.5
        0      0     -1  2.0
        """
        np.testing.assert_equal(percolator.get_indices_below_fdr("Score", fdr_cutoff=0.4), np.array([]))

    def test_get_indices_below_fdr_unordered_idxs(self):
        """Test get_indices_below_fdr."""
        percolator = perc.Percolator(metadata=pd.DataFrame(), input_type="rescore")
        percolator.metrics_val["Score"] = [0, 3, 2, 1, -1]
        percolator.target_decoy_labels = [-1, 1, 1, -1, 1]

        percolator.metrics_val.index = [3, 4, 2, 1, 0]

        """
        idx Score Label       fdr      qval
        4      3      1  0.500000  0.333333
        2      2      1  0.333333  0.333333
        1      1     -1  0.666667  0.600000
        3      0     -1  1.000000  0.600000
        0     -1      1  0.600000  0.600000
        """
        np.testing.assert_equal(percolator.get_indices_below_fdr("Score", fdr_cutoff=0.4), np.array([2, 4]))

    def test_get_indices_below_fdr(self):
        """Test get_indices_below_fdr."""
        percolator = perc.Percolator(metadata=pd.DataFrame(), input_type="rescore")
        percolator.metrics_val["Score"] = [0, 3, 2, 1]
        percolator.target_decoy_labels = [-1, 1, 1, -1]
        """
        idx Score Label       fdr      qval
        1      3      1  0.500000  0.333333
        2      2      1  0.333333  0.333333
        3      1      0  0.666667  0.666667
        0      0      0  1.000000  1.000000
        """
        np.testing.assert_equal(percolator.get_indices_below_fdr("Score", fdr_cutoff=0.4), np.array([1, 2]))

    def test_get_indices_below_fdr_filter_decoy(self):
        """Test get_indices_below_fdr."""
        percolator = perc.Percolator(metadata=pd.DataFrame(), input_type="rescore")
        percolator.metrics_val["Score"] = [0, 3, 2, 1, 4, 5, 6, 7]
        percolator.target_decoy_labels = [-1, 1, 1, -1, -1, 1, 1, 1]
        """
        idx  Score Label      fdr
        7      7      1  0.500000
        6      6      1  0.333333
        5      5      1  0.250000
        4      4      0  0.500000
        1      3      1  0.400000
        2      2      1  0.333333
        3      1      0  0.500000
        0      0      0  0.666667
        """
        np.testing.assert_equal(percolator.get_indices_below_fdr("Score", fdr_cutoff=0.4), np.array([1, 2, 5, 6, 7]))


class TestLda:
    """Class to test lda."""

    def test_apply_lda_and_get_indices_below_fdr(self):
        """Score_2 adds more discriminative power between targets and decoys."""
        percolator = perc.Percolator(metadata=pd.DataFrame(), input_type="rescore")
        percolator.metrics_val["Score"] = [0.0, 3.0, 2.0, 1.0, 4.0, 5.0, 6.0, 7.0]
        percolator.metrics_val["Score_2"] = [1.0, 1.5, 2.0, 1.5, 1.0, 1.5, 2.0, 1.5]
        percolator.target_decoy_labels = np.array([-1, 1, 1, -1, -1, 1, 1, 1])
        """
        idx lda_scores  Label       fdr
        6    8.396540      1  0.500000
        7    4.967968      1  0.333333
        2    4.396540      1  0.250000
        5    2.967968      1  0.200000
        1    0.967968      1  0.166667
        3   -1.032032      0  0.333333
        4   -2.460603      0  0.500000
        0   -6.460603      0  0.666667
        """
        np.testing.assert_equal(
            percolator.apply_lda_and_get_indices_below_fdr(initial_scoring_feature="Score", fdr_cutoff=0.4),
            np.array([1, 2, 5, 6, 7]),
        )


class TestRetentionTimeAlignment(unittest.TestCase):
    """Class to test RT alignment."""

    def _get_aligned_predicted_retention_times_noisy_logistic_error(self, x, y, correct_y, method: str):
        """Test get_aligned_predicted_retention_times for a more realistic, similar to logistic case."""
        aligned_predicted_rts = perc.Percolator.get_aligned_predicted_retention_times(
            y, x, x, curve_fitting_method=method
        )
        np.testing.assert_almost_equal(aligned_predicted_rts, correct_y, decimal=1)

        errors = aligned_predicted_rts - correct_y
        percentile_95 = np.percentile(np.abs(errors), 95)

        max_val = 0.05
        self.assertLess(percentile_95, max_val)

    def _get_aligned_predicted_retention_times_linear_error(
        self, method: str, add_noise: bool = False, shuffle: bool = False
    ):
        """Test get_aligned_predicted_retention_times for linear case."""
        fitting_idx = [0, 3, 5, 6, 9]
        f = lambda x: x / 2 + 1
        n = 10
        dec = 12
        max_score = 1e-12

        observed_rts_all = np.linspace(0, 10, n) * 2
        predicted_rts_all = f(observed_rts_all)

        if add_noise:
            np.random.seed(42)
            observed_rts_all += 0.001 * np.random.random(n)
            dec = 3
            max_score = 1e-3

        observed_rts = observed_rts_all[fitting_idx]
        predicted_rts = predicted_rts_all[fitting_idx]

        if shuffle:
            shuffled_idx = [3, 9, 0, 5, 6]
            predicted_rts_all = predicted_rts_all[shuffled_idx]
            observed_rts_all = observed_rts_all[shuffled_idx]

        aligned_predicted_rts = perc.Percolator.get_aligned_predicted_retention_times(
            observed_rts, predicted_rts, predicted_rts_all, curve_fitting_method="lowess"
        )

        np.testing.assert_almost_equal(aligned_predicted_rts, observed_rts_all, decimal=dec)

        errors = aligned_predicted_rts - observed_rts_all
        percentile_95 = np.percentile(np.abs(errors), 95)

        self.assertLess(percentile_95, max_score)  # , f"95 percentile is not lower than {max_val}")

    def test_linear(self):
        """Test get_aligned_predicted_retention_times for linear case."""
        methods = ["lowess", "spline", "logistic"]
        for method in methods:
            self._get_aligned_predicted_retention_times_linear_error(method)

    def test_linear_with_noise(self):
        """Test get_aligned_predicted_retention_times for linear case with a bit of gaussian noise."""
        methods = ["lowess", "spline", "logistic"]
        for method in methods:
            self._get_aligned_predicted_retention_times_linear_error(method, add_noise=True)

    def test_linear_not_sorted(self):
        """Test get_aligned_predicted_retention_times if idx are not sorted for the query after the fit."""
        methods = ["lowess", "spline", "logistic"]
        for method in methods:
            self._get_aligned_predicted_retention_times_linear_error(method, shuffle=True)

    def test_noisy_logistic(self):
        """Test get_aligned_predicted_retention_times for a more realistic, similar to logistic case."""
        methods = ["spline", "logistic"]
        x, y, correct_y = _create_noisy_logistic_data()
        for method in methods:
            self._get_aligned_predicted_retention_times_noisy_logistic_error(x, y, correct_y, method)

    def test_get_aligned_predicted_retention_times_linear_spline_wrong_method(self):
        """Negative test get_aligned_predicted_retention_times to check if it raises."""
        observed_rts = np.array([])
        predicted_rts = np.array([])
        predicted_rts_all = np.array([])
        self.assertRaises(
            ValueError,
            perc.Percolator.get_aligned_predicted_retention_times,
            observed_rts,
            predicted_rts,
            predicted_rts_all,
            curve_fitting_method="undefined",
        )

    def test_sample_balanced_over_bins(self):
        """Test sample_balanced_over_bins."""
        observed_rts = np.linspace(0, 10, 10) * 2 + 0.001 * np.random.random(10)
        predicted_rts = np.linspace(1, 11, 10)
        retention_time_df = pd.DataFrame()
        retention_time_df["RETENTION_TIME"] = observed_rts
        retention_time_df["PREDICTED_IRT"] = predicted_rts
        sampled_index = perc.Percolator.sample_balanced_over_bins(retention_time_df, sample_size=3)
        np.testing.assert_equal(len(sampled_index), 3)
        np.testing.assert_equal(len(set(sampled_index)), 3)


def _create_noisy_logistic_data():
    """
    Create artifical input to test the functions.

    A logistic function is applied to a range of x values followed by adding noise.
    Then, on the bottom and top of the logistical range, a cloud of noise values is
    generated to make it harder to fit the logistical data manifold.
    The result looks similar to this:

                    . ..   . .   .       ........
               .  .  .   .  ... .  .
        .    .  .      . ..    .
                            .
                          .
                         .
                        .
                       .
                      .
                     .
                    .
                  .
               .     .    .  .. .
          .  .  ... .. .   . .   .
    . .  . .  . ...   .

    The function uses the same seed to make sure the unit tests are reproducable.

    :returns: a tuple of x, y (with noise) and correct_y values
    """
    x = np.linspace(0, 1100, 1100)  # Generate 1100 evenly spaced points between 0 and 1100
    correct_y = 1 / (1 + np.exp(-(x - 500) / 100))  # Compute y using the logistic function
    # Add random noise to y
    np.random.seed(42)  # Set the random seed for reproducibility
    noise = np.random.normal(scale=0.01, size=correct_y.shape)  # some gaussian noise to make it more realistic
    y = correct_y + noise

    # Generate linear data for bottom cloud
    x_linear = np.random.choice(np.arange(700), size=100, replace=False)
    np.random.seed(42)  # Set the random seed for reproducibility
    y_linear = 0.00005 * x_linear + np.random.uniform(low=0, high=0.05, size=len(x_linear)) * 0.003 * x_linear

    # Generate linear data for top cloud
    x_linear2 = np.random.choice(np.arange(200, 900), size=100, replace=False)
    np.random.seed(42)  # Set the random seed for reproducibility
    y_linear2 = (
        0.00005 * x_linear2 + 0.97 - np.random.uniform(low=0, high=0.05, size=len(x_linear2)) / (0.001 * x_linear2)
    )

    # Plot the data
    y[x_linear] = y_linear
    y[x_linear2] = y_linear2
    return x, y, correct_y


class TestPercolator:
    """Class to test percolator."""

    def test_get_specid(self):
        """Test get_specid."""
        np.testing.assert_string_equal(
            perc.Percolator.get_specid(("rawfile", 1234, "ABCD", 2, 1)), "rawfile-1234-ABCD-2-1"
        )

    def test_count_missed_cleavages(self):
        """Test count_missed_cleavages."""
        np.testing.assert_equal(perc.Percolator.count_missed_cleavages("AKAAAAKAK"), 2)

    def test_count_arginines_and_lysines(self):
        """Test count_arginines_and_lysines."""
        np.testing.assert_equal(perc.Percolator.count_arginines_and_lysines("ARAAAAKAK"), 3)

    def test_calculate_mass_difference(self):
        """Test calculate_mass_difference."""
        np.testing.assert_almost_equal(perc.Percolator.calculate_mass_difference((1000.0, 1001.2)), 1.2)

    def test_calculate_mass_difference_ppm(self):
        """Test calculate_mass_difference_ppm."""
        np.testing.assert_almost_equal(perc.Percolator.calculate_mass_difference_ppm((1000.0, 1001.2)), 1200.0)

    def test_get_target_decoy_label_target(self):
        """Test get_target_decoy_label_target."""
        reverse = False
        np.testing.assert_equal(perc.Percolator.get_target_decoy_label(reverse), 1)

    def test_get_target_decoy_label_decoy(self):
        """Test get_target_decoy_label_decoy."""
        reverse = True
        np.testing.assert_equal(perc.Percolator.get_target_decoy_label(reverse), -1)

    def test_get_delta_score(self):
        """Test get_delta_score."""
        df = pd.DataFrame()
        df["spectral_angle"] = [100, 80, 40, 50, 300, 10]
        df["ScanNr"] = [1, 1, 1, 2, 2, 2]
        np.testing.assert_equal(
            perc.Percolator.get_delta_score(df, "spectral_angle"), np.array([20, 40, 0, 40, 250, 0])
        )

    def test_add_additional_features(self):
        """Test add_additional_features."""
        types = {
            "RAW_FILE": str,
            "SCAN_NUMBER": int,
            "MODIFIED_SEQUENCE": str,
            "PRECURSOR_CHARGE": int,
            "SCAN_EVENT_NUMBER": int,
            "MASS": float,
            "SCORE": float,
            "REVERSE": int,
            "SEQUENCE": str,
            "PEPTIDE_LENGTH": float,
            "A": float,
            "B": float,
            "precursor_charge": float,
            "Unnamed 1": int,
        }

        perc_input = pd.DataFrame(columns=types).astype(types)

        percolator_all = perc.Percolator(metadata=perc_input, input_type="rescore", additional_columns="all")
        percolator_all.add_additional_features()
        pd.testing.assert_frame_equal(
            pd.DataFrame(columns=["A", "B"]).astype({"A": float, "B": float}), percolator_all.metrics_val
        )

        percolator_list = perc.Percolator(metadata=perc_input, input_type="rescore", additional_columns=["A"])
        percolator_list.add_additional_features()
        pd.testing.assert_frame_equal(pd.DataFrame(columns=["A"]).astype({"A": float}), percolator_list.metrics_val)

        percolator_none = perc.Percolator(metadata=perc_input, input_type="rescore", additional_columns="none")
        percolator_none.add_additional_features()
        pd.testing.assert_frame_equal(pd.DataFrame(), percolator_none.metrics_val)

    def test_calc(self):
        """Test calc."""
        perc_input = pd.read_csv(Path(__file__).parent / "data/perc_input.csv")
        z = constants.EPSILON
        #                                         y1.1  y1.2  y1.3  b1.1  b1.2  b1.3  y2.1  y2.2  y2.3
        predicted_intensities_target = get_padded_array([7.2, 2.3, 0.01, 0.02, 6.1, 3.1, z, z, 0])
        observed_intensities_target = get_padded_array([10.2, z, 1.3, z, 8.2, z, 3.2, z, 0])
        mz_target = get_padded_array([100, 0, 150, 0, 0, 0, 300, 0, 0])

        predicted_intensities_decoy = get_padded_array([z, 3.0, 4.0, z])
        observed_intensities_decoy = get_padded_array([z, z, 3.0, 4.0])
        mz_decoy = get_padded_array([0, 0, 100, 0])

        predicted_intensities = scipy.sparse.vstack(np.repeat(predicted_intensities_target, len(perc_input)))
        observed_intensities = scipy.sparse.vstack(np.repeat(observed_intensities_target, len(perc_input)))
        mz = scipy.sparse.vstack(np.repeat(mz_target, len(perc_input)))

        predicted_intensities[1, :] = predicted_intensities_decoy
        predicted_intensities[2, :] = predicted_intensities_decoy
        observed_intensities[1, :] = observed_intensities_decoy
        observed_intensities[2, :] = observed_intensities_decoy
        mz[1, :] = mz_decoy
        mz[2, :] = mz_target

        percolator = perc.Percolator(
            metadata=perc_input,
            input_type="rescore",
            pred_intensities=predicted_intensities,
            true_intensities=observed_intensities,
            mz=mz,
            fdr_cutoff=0.4,
            regression_method="lowess",
        )
        percolator.calc()

        # meta data for percolator
        np.testing.assert_string_equal(
            percolator.metrics_val["SpecId"][0], "20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02-7978-AAIGEATRL-2-1"
        )
        np.testing.assert_equal(percolator.metrics_val["Label"][0], 1)
        np.testing.assert_equal(percolator.metrics_val["ScanNr"][0], 7978)
        np.testing.assert_equal(percolator.metrics_val["filename"][0], "20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02")
        # np.testing.assert_almost_equal(percolator.metrics_val['ExpMass'][0], 900.50345678)
        np.testing.assert_string_equal(percolator.metrics_val["Peptide"][0], "_.AAIGEATRL._")
        np.testing.assert_string_equal(
            percolator.metrics_val["Proteins"][0], "sp|O23523|RGGA_ARATH"
        )  # we don't need the protein ID to get PSM / peptide results

        # features
        np.testing.assert_equal(percolator.metrics_val["missedCleavages"][0], 1)
        np.testing.assert_equal(percolator.metrics_val["KR"][0], 1)
        np.testing.assert_equal(percolator.metrics_val["sequence_length"][0], 9)
        np.testing.assert_almost_equal(
            percolator.metrics_val["Mass"][0], 900.50288029264
        )  # this is the calculated mass as a feature
        # np.testing.assert_almost_equal(percolator.metrics_val['deltaM_Da'][0], -0.0005764873)
        # np.testing.assert_almost_equal(percolator.metrics_val['absDeltaM_Da'][0], 0.0005764873)
        # np.testing.assert_almost_equal(percolator.metrics_val['deltaM_ppm'][0], -0.64018339472)
        # np.testing.assert_almost_equal(percolator.metrics_val['absDeltaM_ppm'][0], 0.64018339472)
        np.testing.assert_equal(percolator.metrics_val["Charge2"][0], 1)
        np.testing.assert_equal(percolator.metrics_val["Charge3"][0], 0)
        np.testing.assert_equal(percolator.metrics_val["UnknownFragmentationMethod"][0], 0)
        np.testing.assert_equal(percolator.metrics_val["HCD"][0], 1)
        np.testing.assert_equal(percolator.metrics_val["CID"][0], 0)

        np.testing.assert_almost_equal(percolator.metrics_val["RT"][0], 0.5, decimal=3)
        np.testing.assert_almost_equal(percolator.metrics_val["pred_RT"][0], 0.5, decimal=3)
        np.testing.assert_almost_equal(percolator.metrics_val["abs_rt_diff"][0], 0.0, decimal=3)

        # check label of second PSM (decoy)
        np.testing.assert_equal(percolator.metrics_val["Label"][1], -1)

        # check lowess fit of second PSM
        np.testing.assert_almost_equal(percolator.metrics_val["abs_rt_diff"][1], 0.0, decimal=3)
        np.testing.assert_almost_equal(percolator.metrics_val["abs_rt_diff"][2], 0.0, decimal=3)
        # TODO: only add this feature if they are not all zero
        # np.testing.assert_equal(percolator.metrics_val['spectral_angle_delta_score'][0], 0.0)

    def test_calc_all_features(self):
        """Test calc."""
        perc_input = pd.read_csv(Path(__file__).parent / "data/perc_input.csv")
        z = constants.EPSILON
        #                                         y1.1  y1.2  y1.3  b1.1  b1.2  b1.3  y2.1  y2.2  y2.3
        predicted_intensities_target = get_padded_array([7.2, 2.3, 0.01, 0.02, 6.1, 3.1, z, z, 0])
        observed_intensities_target = get_padded_array([10.2, z, 1.3, z, 8.2, z, 3.2, z, 0])
        mz_target = get_padded_array([100, 0, 150, 0, 0, 0, 300, 0, 0])

        predicted_intensities_decoy = get_padded_array([z, 3.0, 4.0, z])
        observed_intensities_decoy = get_padded_array([z, z, 3.0, 4.0])
        mz_decoy = get_padded_array([0, 0, 100, 0])

        predicted_intensities = scipy.sparse.vstack(np.repeat(predicted_intensities_target, len(perc_input)))
        observed_intensities = scipy.sparse.vstack(np.repeat(observed_intensities_target, len(perc_input)))
        mz = scipy.sparse.vstack(np.repeat(mz_target, len(perc_input)))

        predicted_intensities[1, :] = predicted_intensities_decoy
        predicted_intensities[2, :] = predicted_intensities_decoy
        observed_intensities[1, :] = observed_intensities_decoy
        observed_intensities[2, :] = observed_intensities_decoy
        mz[1, :] = mz_decoy
        mz[2, :] = mz_target

        percolator = perc.Percolator(
            metadata=perc_input,
            input_type="rescore",
            pred_intensities=predicted_intensities,
            true_intensities=observed_intensities,
            mz=mz,
            fdr_cutoff=0.4,
            regression_method="lowess",
            all_features_flag=True,
        )
        percolator.calc()


def get_padded_array(arr, padding_value=0):
    """Get padded array."""
    return scipy.sparse.csr_matrix(
        np.array([np.pad(arr, (0, constants.VEC_LENGTH - len(arr)), "constant", constant_values=padding_value)])
    )
