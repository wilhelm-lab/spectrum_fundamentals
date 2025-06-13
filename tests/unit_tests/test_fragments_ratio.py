import numpy as np
import scipy.sparse

import spectrum_fundamentals.constants as constants
import spectrum_fundamentals.metrics.fragments_ratio as fr


class TestObservationState:
    """Class to test observation state."""

    def test_get_mask_observed_valid(self):
        """Test get_mask_observed_valid."""
        observed_mz = get_padded_array([10.2, constants.EPSILON, 0, 0.0])
        assert_equal_sparse(
            fr.FragmentsRatio.get_mask_observed_valid(observed_mz), get_padded_array([True, True, False, False])
        )

    def test_make_boolean(self):
        """Test make_boolean."""
        observed = get_padded_array([10.2, constants.EPSILON, 0, 0.0])
        mask = get_padded_array([True, True, False, False])
        assert_equal_sparse(
            fr.FragmentsRatio.make_boolean(observed, mask), get_padded_array([True, False, False, False])
        )

    def test_make_boolean_cutoff_below(self):
        """Test make_boolean_cutoff_below."""
        predicted = get_padded_array([10.2, constants.EPSILON, 0, 0.02, 0.0])
        mask = get_padded_array([True, True, False, True, False])
        assert_equal_sparse(
            fr.FragmentsRatio.make_boolean(predicted, mask, cutoff=0.05),
            get_padded_array([True, False, False, False, False]),
        )

    def test_make_boolean_cutoff_above(self):
        """Test make_boolean_cutoff_above."""
        predicted = get_padded_array([10.2, constants.EPSILON, 0, 0.02, 0.0])
        mask = get_padded_array([True, True, False, True, False])
        assert_equal_sparse(
            fr.FragmentsRatio.make_boolean(predicted, mask, cutoff=0.01),
            get_padded_array([True, False, False, True, False]),
        )

    def test_get_observation_state(self):
        """Test get_observation_state."""
        observed_boolean = get_padded_array([False, False, True, True])
        predicted_boolean = get_padded_array([False, True, False, True])
        mask = get_padded_array([True, True, True, True])
        assert_equal_sparse(
            fr.FragmentsRatio.get_observation_state(observed_boolean, predicted_boolean, mask),
            get_padded_array(
                [
                    fr.ObservationState.NOT_OBS_AND_NOT_PRED,
                    fr.ObservationState.NOT_OBS_BUT_PRED,
                    fr.ObservationState.OBS_BUT_NOT_PRED,
                    fr.ObservationState.OBS_AND_PRED,
                ],
                fr.ObservationState.INVALID_ION,
            ),
        )


class TestCountIons:
    """Class to test ion counts."""

    def test_count_ions(self):
        """Test count ions predicted."""
        predicted_boolean = get_padded_array([True, False, False, False, False])
        np.testing.assert_equal(fr.FragmentsRatio.count_with_ion_mask(predicted_boolean), 1)

    def test_count_ions_b(self):
        """Test count b ions predicted."""
        predicted_boolean = get_padded_array([True, False, False, False, False])
        np.testing.assert_equal(fr.FragmentsRatio.count_with_ion_mask(predicted_boolean, constants.B_ION_MASK), 0)

    def test_count_ions_y(self):
        """Test count y ions predicted."""
        predicted_boolean = get_padded_array([True, False, False, False, False])
        np.testing.assert_equal(fr.FragmentsRatio.count_with_ion_mask(predicted_boolean, constants.Y_ION_MASK), 1)


class TestCountObservedAndPredicted:
    """Class to test count observed and predicted."""

    def test_count_observed_and_predicted(self):
        """Test count observation states observed and predicted."""
        observation_state = get_padded_array(
            [
                fr.ObservationState.NOT_OBS_AND_NOT_PRED,
                fr.ObservationState.NOT_OBS_BUT_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
                fr.ObservationState.OBS_AND_PRED,
            ],
            fr.ObservationState.INVALID_ION,
        )
        np.testing.assert_equal(
            fr.FragmentsRatio.count_observation_states(observation_state, fr.ObservationState.OBS_AND_PRED), 1
        )

    def test_count_observed_and_predicted_b(self):
        """Test count observation states observed and predicted b ions."""
        observation_state = get_padded_array(
            [
                fr.ObservationState.NOT_OBS_AND_NOT_PRED,
                fr.ObservationState.NOT_OBS_BUT_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
                fr.ObservationState.OBS_AND_PRED,
            ],
            fr.ObservationState.INVALID_ION,
        )
        np.testing.assert_equal(
            fr.FragmentsRatio.count_observation_states(
                observation_state, fr.ObservationState.OBS_AND_PRED, constants.B_ION_MASK
            ),
            1,
        )

    def test_count_observed_and_predicted_y(self):
        """Test count observation states observed and predicted y ions."""
        observation_state = get_padded_array(
            [
                fr.ObservationState.NOT_OBS_AND_NOT_PRED,
                fr.ObservationState.NOT_OBS_BUT_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
                fr.ObservationState.OBS_AND_PRED,
            ],
            fr.ObservationState.INVALID_ION,
        )
        np.testing.assert_equal(
            fr.FragmentsRatio.count_observation_states(
                observation_state, fr.ObservationState.OBS_AND_PRED, constants.Y_ION_MASK
            ),
            0,
        )


class TestCountNotObservedAndNotPredicted:
    """Class to test count not observed and not predicted."""

    def test_count_not_observed_and_not_predicted(self):
        """Test count observation states not observed and not predicted."""
        observation_state = get_padded_array(
            [
                fr.ObservationState.NOT_OBS_AND_NOT_PRED,
                fr.ObservationState.NOT_OBS_BUT_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
                fr.ObservationState.OBS_AND_PRED,
                fr.ObservationState.NOT_OBS_AND_NOT_PRED,
            ],
            fr.ObservationState.INVALID_ION,
        )
        np.testing.assert_equal(
            fr.FragmentsRatio.count_observation_states(observation_state, fr.ObservationState.NOT_OBS_AND_NOT_PRED), 2
        )

    def test_count_not_observed_and_not_predicted_b(self):
        """Test count observation states not observed and not predicted b ions."""
        observation_state = get_padded_array(
            [
                fr.ObservationState.NOT_OBS_AND_NOT_PRED,
                fr.ObservationState.NOT_OBS_BUT_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
                fr.ObservationState.OBS_AND_PRED,
                fr.ObservationState.NOT_OBS_AND_NOT_PRED,
            ],
            fr.ObservationState.INVALID_ION,
        )
        np.testing.assert_equal(
            fr.FragmentsRatio.count_observation_states(
                observation_state, fr.ObservationState.NOT_OBS_AND_NOT_PRED, constants.B_ION_MASK
            ),
            1,
        )

    def test_count_not_observed_and_not_predicted_y(self):
        """Test count observation states not observed and not predicted y ions."""
        observation_state = get_padded_array(
            [
                fr.ObservationState.NOT_OBS_AND_NOT_PRED,
                fr.ObservationState.NOT_OBS_BUT_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
                fr.ObservationState.OBS_AND_PRED,
                fr.ObservationState.NOT_OBS_AND_NOT_PRED,
            ],
            fr.ObservationState.INVALID_ION,
        )
        np.testing.assert_equal(
            fr.FragmentsRatio.count_observation_states(
                observation_state, fr.ObservationState.NOT_OBS_AND_NOT_PRED, constants.Y_ION_MASK
            ),
            1,
        )


class TestCountNotObservedButPredicted:
    """Class to test count not observed but predicted."""

    def test_count_not_observed_but_predicted(self):
        """Test count observation states not observed but predicted."""
        observation_state = get_padded_array(
            [
                fr.ObservationState.NOT_OBS_AND_NOT_PRED,
                fr.ObservationState.NOT_OBS_BUT_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
                fr.ObservationState.OBS_AND_PRED,
                fr.ObservationState.NOT_OBS_BUT_PRED,
                fr.ObservationState.NOT_OBS_BUT_PRED,
            ],
            fr.ObservationState.INVALID_ION,
        )
        np.testing.assert_equal(
            fr.FragmentsRatio.count_observation_states(observation_state, fr.ObservationState.NOT_OBS_BUT_PRED), 3
        )

    def test_count_not_observed_but_predicted_b(self):
        """Test count observation states not observed but predicted b ions."""
        observation_state = get_padded_array(
            [
                fr.ObservationState.NOT_OBS_AND_NOT_PRED,
                fr.ObservationState.NOT_OBS_BUT_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
                fr.ObservationState.OBS_AND_PRED,
                fr.ObservationState.NOT_OBS_BUT_PRED,
                fr.ObservationState.NOT_OBS_BUT_PRED,
            ],
            fr.ObservationState.INVALID_ION,
        )
        np.testing.assert_equal(
            fr.FragmentsRatio.count_observation_states(
                observation_state, fr.ObservationState.NOT_OBS_BUT_PRED, constants.B_ION_MASK
            ),
            2,
        )

    def test_count_not_observed_but_predicted_y(self):
        """Test count observation states not observed but predicted y ions."""
        observation_state = get_padded_array(
            [
                fr.ObservationState.NOT_OBS_AND_NOT_PRED,
                fr.ObservationState.NOT_OBS_BUT_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
                fr.ObservationState.OBS_AND_PRED,
                fr.ObservationState.NOT_OBS_BUT_PRED,
                fr.ObservationState.NOT_OBS_BUT_PRED,
            ],
            fr.ObservationState.INVALID_ION,
        )
        np.testing.assert_equal(
            fr.FragmentsRatio.count_observation_states(
                observation_state, fr.ObservationState.NOT_OBS_BUT_PRED, constants.Y_ION_MASK
            ),
            1,
        )


class TestCountObservedButNotPredicted:
    """Class to test count observed but not predicted."""

    def test_count_observed_but_not_predicted(self):
        """Test count observation states observed but not predicted."""
        observation_state = get_padded_array(
            [
                fr.ObservationState.NOT_OBS_AND_NOT_PRED,
                fr.ObservationState.NOT_OBS_BUT_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
                fr.ObservationState.OBS_AND_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
            ],
            fr.ObservationState.INVALID_ION,
        )
        np.testing.assert_equal(
            fr.FragmentsRatio.count_observation_states(observation_state, fr.ObservationState.OBS_BUT_NOT_PRED), 4
        )

    def test_count_observed_but_not_predicted_b(self):
        """Test count observation states observed but not predicted b ions."""
        observation_state = get_padded_array(
            [
                fr.ObservationState.NOT_OBS_AND_NOT_PRED,
                fr.ObservationState.NOT_OBS_BUT_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
                fr.ObservationState.OBS_AND_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
            ],
            fr.ObservationState.INVALID_ION,
        )
        np.testing.assert_equal(
            fr.FragmentsRatio.count_observation_states(
                observation_state, fr.ObservationState.OBS_BUT_NOT_PRED, constants.B_ION_MASK
            ),
            2,
        )

    def test_count_observed_but_not_predicted_y(self):
        """Test count observation states observed but not predicted y ions."""
        observation_state = get_padded_array(
            [
                fr.ObservationState.NOT_OBS_AND_NOT_PRED,
                fr.ObservationState.NOT_OBS_BUT_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
                fr.ObservationState.OBS_AND_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
                fr.ObservationState.OBS_BUT_NOT_PRED,
            ],
            fr.ObservationState.INVALID_ION,
        )

        np.testing.assert_equal(
            fr.FragmentsRatio.count_observation_states(
                observation_state, fr.ObservationState.OBS_BUT_NOT_PRED, constants.Y_ION_MASK
            ),
            2,
        )


class TestCalc:
    """Class to test calc."""

    def test_calc_xl(self):
        """Test calc."""
        z = constants.EPSILON
        #                                         y1.1  y1.2  y1.3  b1.1  b1.2  b1.3  y2.1  y2.2  y2.3
        predicted_intensities = get_padded_array([7.2, 2.3, 0.01, 0.02, 6.1, 3.1, z, z, 0], xl=True)
        observed_intensities = get_padded_array([10.2, z, 1.3, z, 8.2, z, 3.2, z, 0], xl=True)

        fragments_ratio = fr.FragmentsRatio(predicted_intensities, observed_intensities)
        fragments_ratio.calc(xl=True, cms2=True)

        # counting metrics - peptide_a
        np.testing.assert_equal(fragments_ratio.metrics_val["count_predicted_a"][0], 4)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_predicted_b_a"][0], 2)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_predicted_y_a"][0], 2)

        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_a"][0], 4)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_b_a"][0], 1)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_y_a"][0], 3)

        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_and_predicted_a"][0], 2)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_and_predicted_b_a"][0], 1)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_and_predicted_y_a"][0], 1)

        np.testing.assert_equal(fragments_ratio.metrics_val["count_not_observed_and_not_predicted_a"][0], 2)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_not_observed_and_not_predicted_b_a"][0], 1)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_not_observed_and_not_predicted_y_a"][0], 1)

        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_but_not_predicted_a"][0], 2)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_but_not_predicted_b_a"][0], 0)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_but_not_predicted_y_a"][0], 2)

        np.testing.assert_equal(fragments_ratio.metrics_val["count_not_observed_but_predicted_a"][0], 2)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_not_observed_but_predicted_b_a"][0], 1)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_not_observed_but_predicted_y_a"][0], 1)

        # fractional count metrics - peptide a
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_predicted_a"][0], 4 / 8)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_predicted_b_a"][0], 2 / 3)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_predicted_y_a"][0], 2 / 5)

        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_a"][0], 4 / 8)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_b_a"][0], 1 / 3)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_y_a"][0], 3 / 5)

        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_and_predicted_a"][0], 2 / 8)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_and_predicted_b_a"][0], 1 / 3)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_and_predicted_y_a"][0], 1 / 5)

        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_not_observed_and_not_predicted_a"][0], 2 / 8)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_not_observed_and_not_predicted_b_a"][0], 1 / 3)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_not_observed_and_not_predicted_y_a"][0], 1 / 5)

        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_but_not_predicted_a"][0], 2 / 8)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_but_not_predicted_b_a"][0], 0 / 3)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_but_not_predicted_y_a"][0], 2 / 5)

        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_not_observed_but_predicted_a"][0], 2 / 8)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_not_observed_but_predicted_b_a"][0], 1 / 3)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_not_observed_but_predicted_y_a"][0], 1 / 5)

        # fractional count metrics relative to predictions - peptide a
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_and_predicted_vs_predicted_a"][0], 2 / 4)
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_observed_and_predicted_b_vs_predicted_b_a"][0], 1 / 2
        )
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_observed_and_predicted_y_vs_predicted_y_a"][0], 1 / 2
        )
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_not_observed_and_not_predicted_vs_predicted_a"][0], 2 / 4
        )
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_not_observed_and_not_predicted_b_vs_predicted_b_a"][0], 1 / 2
        )
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_not_observed_and_not_predicted_y_vs_predicted_y_a"][0], 1 / 2
        )
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_observed_but_not_predicted_vs_predicted_a"][0], 2 / 4
        )
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_observed_but_not_predicted_b_vs_predicted_b_a"][0], 0 / 2
        )
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_observed_but_not_predicted_y_vs_predicted_y_a"][0], 2 / 2
        )

        # peptide  b

        np.testing.assert_equal(fragments_ratio.metrics_val["count_predicted_b"][0], 4)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_predicted_b_b"][0], 2)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_predicted_y_b"][0], 2)

        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_b"][0], 4)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_b_b"][0], 1)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_y_b"][0], 3)

        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_and_predicted_b"][0], 2)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_and_predicted_b_b"][0], 1)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_and_predicted_y_b"][0], 1)

        np.testing.assert_equal(fragments_ratio.metrics_val["count_not_observed_and_not_predicted_b"][0], 2)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_not_observed_and_not_predicted_b_b"][0], 1)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_not_observed_and_not_predicted_y_b"][0], 1)

        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_but_not_predicted_b"][0], 2)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_but_not_predicted_b_b"][0], 0)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_but_not_predicted_y_b"][0], 2)

        np.testing.assert_equal(fragments_ratio.metrics_val["count_not_observed_but_predicted_b"][0], 2)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_not_observed_but_predicted_b_b"][0], 1)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_not_observed_but_predicted_y_b"][0], 1)

        # fractional count metrics - peptide b
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_predicted_b"][0], 4 / 8)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_predicted_b_b"][0], 2 / 3)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_predicted_y_b"][0], 2 / 5)

        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_b"][0], 4 / 8)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_b_b"][0], 1 / 3)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_y_b"][0], 3 / 5)

        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_and_predicted_b"][0], 2 / 8)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_and_predicted_b_b"][0], 1 / 3)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_and_predicted_y_b"][0], 1 / 5)

        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_not_observed_and_not_predicted_b"][0], 2 / 8)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_not_observed_and_not_predicted_b_b"][0], 1 / 3)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_not_observed_and_not_predicted_y_b"][0], 1 / 5)

        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_but_not_predicted_b"][0], 2 / 8)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_but_not_predicted_b_b"][0], 0 / 3)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_but_not_predicted_y_b"][0], 2 / 5)

        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_not_observed_but_predicted_b"][0], 2 / 8)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_not_observed_but_predicted_b_b"][0], 1 / 3)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_not_observed_but_predicted_y_b"][0], 1 / 5)

        # fractional count metrics relative to predictions - peptide b
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_and_predicted_vs_predicted_b"][0], 2 / 4)
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_observed_and_predicted_b_vs_predicted_b_b"][0], 1 / 2
        )
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_observed_and_predicted_y_vs_predicted_y_b"][0], 1 / 2
        )
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_not_observed_and_not_predicted_vs_predicted_b"][0], 2 / 4
        )
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_not_observed_and_not_predicted_b_vs_predicted_b_b"][0], 1 / 2
        )
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_not_observed_and_not_predicted_y_vs_predicted_y_b"][0], 1 / 2
        )
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_observed_but_not_predicted_vs_predicted_b"][0], 2 / 4
        )
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_observed_but_not_predicted_b_vs_predicted_b_b"][0], 0 / 2
        )
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_observed_but_not_predicted_y_vs_predicted_y_b"][0], 2 / 2
        )

    def test_calc(self):
        """Test calc."""
        z = constants.EPSILON
        #                                         y1.1  y1.2  y1.3  b1.1  b1.2  b1.3  y2.1  y2.2  y2.3
        predicted_intensities = get_padded_array([7.2, 2.3, 0.01, 0.02, 6.1, 3.1, z, z, 0])
        observed_intensities = get_padded_array([10.2, z, 1.3, z, 8.2, z, 3.2, z, 0])
        fragments_ratio = fr.FragmentsRatio(predicted_intensities, observed_intensities)
        fragments_ratio.calc()

        # counting metrics
        np.testing.assert_equal(fragments_ratio.metrics_val["count_predicted"][0], 4)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_predicted_b"][0], 2)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_predicted_y"][0], 2)

        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed"][0], 4)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_b"][0], 1)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_y"][0], 3)

        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_and_predicted"][0], 2)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_and_predicted_b"][0], 1)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_and_predicted_y"][0], 1)

        np.testing.assert_equal(fragments_ratio.metrics_val["count_not_observed_and_not_predicted"][0], 2)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_not_observed_and_not_predicted_b"][0], 1)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_not_observed_and_not_predicted_y"][0], 1)

        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_but_not_predicted"][0], 2)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_but_not_predicted_b"][0], 0)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_observed_but_not_predicted_y"][0], 2)

        np.testing.assert_equal(fragments_ratio.metrics_val["count_not_observed_but_predicted"][0], 2)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_not_observed_but_predicted_b"][0], 1)
        np.testing.assert_equal(fragments_ratio.metrics_val["count_not_observed_but_predicted_y"][0], 1)

        # fractional count metrics
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_predicted"][0], 4 / 8)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_predicted_b"][0], 2 / 3)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_predicted_y"][0], 2 / 5)

        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed"][0], 4 / 8)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_b"][0], 1 / 3)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_y"][0], 3 / 5)

        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_and_predicted"][0], 2 / 8)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_and_predicted_b"][0], 1 / 3)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_and_predicted_y"][0], 1 / 5)

        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_not_observed_and_not_predicted"][0], 2 / 8)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_not_observed_and_not_predicted_b"][0], 1 / 3)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_not_observed_and_not_predicted_y"][0], 1 / 5)

        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_but_not_predicted"][0], 2 / 8)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_but_not_predicted_b"][0], 0 / 3)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_but_not_predicted_y"][0], 2 / 5)

        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_not_observed_but_predicted"][0], 2 / 8)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_not_observed_but_predicted_b"][0], 1 / 3)
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_not_observed_but_predicted_y"][0], 1 / 5)

        # fractional count metrics relative to predictions
        np.testing.assert_equal(fragments_ratio.metrics_val["fraction_observed_and_predicted_vs_predicted"][0], 2 / 4)
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_observed_and_predicted_b_vs_predicted_b"][0], 1 / 2
        )
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_observed_and_predicted_y_vs_predicted_y"][0], 1 / 2
        )

        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_not_observed_and_not_predicted_vs_predicted"][0], 2 / 4
        )
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_not_observed_and_not_predicted_b_vs_predicted_b"][0], 1 / 2
        )
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_not_observed_and_not_predicted_y_vs_predicted_y"][0], 1 / 2
        )

        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_observed_but_not_predicted_vs_predicted"][0], 2 / 4
        )
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_observed_but_not_predicted_b_vs_predicted_b"][0], 0 / 2
        )
        np.testing.assert_equal(
            fragments_ratio.metrics_val["fraction_observed_but_not_predicted_y_vs_predicted_y"][0], 2 / 2
        )

        # np.testing.assert_equal(fragmentsRatio.metrics_val['fraction_not_observed_but_predicted_vs_predicted'][0], 2 / 4)
        # np.testing.assert_equal(fragmentsRatio.metrics_val['fraction_not_observed_but_predicted_b_vs_predicted_b'][0], 1 / 2)
        # np.testing.assert_equal(fragmentsRatio.metrics_val['fraction_not_observed_but_predicted_y_vs_predicted_y'][0], 1 / 2)


def assert_equal_sparse(a, b):
    """Check that there are 0 elements for which a != b."""
    assert (a != b).nnz == 0  # checks that there are 0 elements for which a != b


def get_padded_array(arr, padding_value=0, xl: bool = False):
    """Get padded array."""
    if xl:
        padded_arr = np.pad(arr, (0, constants.VEC_LENGTH_CMS2 - len(arr)), "constant", constant_values=padding_value)
        return scipy.sparse.csr_matrix(np.concatenate([padded_arr, padded_arr]))

    else:
        return scipy.sparse.csr_matrix(
            np.array([np.pad(arr, (0, constants.VEC_LENGTH - len(arr)), "constant", constant_values=padding_value)])
        )
