import enum
from typing import Optional, Union

import numpy as np
import scipy.sparse

from .. import constants
from .metric import Metric


class ObservationState(enum.IntEnum):
    """
    States.

    - 4: not seen in either
    - 3: predicted but not in observed
    - 2: seen in both
    - 1: observed but not in predicted
    - 0: invalid
    """

    NOT_OBS_AND_NOT_PRED = 4
    NOT_OBS_BUT_PRED = 3
    OBS_AND_PRED = 2
    OBS_BUT_NOT_PRED = 1
    INVALID_ION = 0


class FragmentsRatio(Metric):
    """Main to initialize a FragmentsRatio obj."""

    @staticmethod
    def count_with_ion_mask(
        boolean_array: scipy.sparse.csr_matrix,
        ion_mask: Optional[Union[np.ndarray, scipy.sparse.spmatrix]] = None,
        xl: bool = False,
    ) -> np.ndarray:
        """
        Count the number of ions.

        :param boolean_array: boolean array with True for observed/predicted peaks and \
                              False for missing observed/predicted peaks, array of length 174
        :param ion_mask: mask with 1s for the ions that should be counted and 0s for ions that should be ignored, \
                         integer array of length 174 for linear and 348 for crosslinked peptides, or a list of integers,
                         or a scipy.sparse.csr_matrix or scipy.sparse._csc.csc_matrix.
        :param xl: whether to process with crosslinked or linear peptides
        :return: number of observed/predicted peaks not masked by ion_mask
        """
        if xl:
            array_size = 348
        else:
            array_size = 174

        if ion_mask is None:
            ion_mask = scipy.sparse.csr_matrix(np.ones((array_size, 1)))
        else:
            ion_mask = scipy.sparse.csr_matrix(ion_mask).T
        return scipy.sparse.csr_matrix.dot(boolean_array, ion_mask).toarray().flatten()

    @staticmethod
    def count_observation_states(
        observation_state: scipy.sparse.csr_matrix,
        test_state: int,
        ion_mask: Optional[Union[np.ndarray, scipy.sparse.csr_matrix]] = None,
        xl: bool = False,
    ) -> np.ndarray:
        """
        Count the number of observation states.

        :param observation_state: integer observation_state, array of length 174
        :param test_state: integer for the test observation state
        :param ion_mask: mask with 1s for the ions that should be counted and 0s for ions that should be ignored, \
                         integer array of length 174
        :param xl: whether or not the function is executed with xl mode
        :return: number of observation states equal to test_state per row
        """
        state_boolean = observation_state == test_state
        return FragmentsRatio.count_with_ion_mask(state_boolean, ion_mask, xl=xl)

    @staticmethod
    def get_mask_observed_valid(observed_mz: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
        """
        Creates a mask out of an observed m/z array with True for invalid entries and \
        False for valid entries in the observed intensities array.

        :param observed_mz: observed m/z, array of length 174
        :return: boolean array, array of length 174
        """
        return observed_mz > 0

    @staticmethod
    def make_boolean(
        intensities: scipy.sparse.csr_matrix, mask: scipy.sparse.csr_matrix, cutoff: float = 2e-7
    ) -> scipy.sparse.csr_matrix:
        """
        Transform array of intensities into boolean array with True if > cutoff and False otherwise.

        :param intensities: observed or predicted intensities, array of length 174
        :param mask: mask with True for invalid values in the observed intensities array, boolean array of length 174
        :param cutoff: minimum intensity value to be considered a peak, for observed intensities use the default cutoff of 0.0, \
                       for predicted intensities, set a cutoff, e.g. 0.05
        :return: boolean array, array of length 174
        """
        intensities_above_cutoff = (intensities > cutoff).multiply(mask)
        return intensities_above_cutoff

    @staticmethod
    def get_observation_state(
        observed_boolean: scipy.sparse.csr_matrix,
        predicted_boolean: scipy.sparse.csr_matrix,
        mask: scipy.sparse.csr_matrix,
    ) -> scipy.sparse.csr_matrix:
        """
        Computes the observation state between the observed and predicted boolean arrays.

        possible values:
        - 4: not seen in either
        - 3: predicted but not in observed
        - 2: seen in both
        - 1: observed but not in predicted
        - 0: invalid
        :param observed_boolean: boolean observed intensities, boolean array of length 174
        :param predicted_boolean : boolean predicted intensities, boolean array of length 174
        :param mask: mask with True for invalid values in the observed intensities array, boolean array of length 174
        :return: integer array, array of length 174
        """
        if scipy.sparse.issparse(observed_boolean):
            observation_state = scipy.sparse.csr_matrix(observed_boolean.shape, dtype=int)
        else:
            observation_state = np.zeros_like(observed_boolean, dtype=int)
        observation_state += observed_boolean.multiply(predicted_boolean) * int(ObservationState.OBS_AND_PRED)
        observation_state += (observed_boolean > predicted_boolean) * int(ObservationState.OBS_BUT_NOT_PRED)
        observation_state += (observed_boolean < predicted_boolean) * int(ObservationState.NOT_OBS_BUT_PRED)
        observation_state += (mask > (observed_boolean + predicted_boolean)) * int(
            ObservationState.NOT_OBS_AND_NOT_PRED
        )
        return observation_state

    def calc(self, xl: bool = False):
        """Adds columns with count, fraction and fraction_predicted features to metrics_val dataframe."""
        if self.true_intensities is None or self.pred_intensities is None:
            return None
        if xl:
            true_intensities_a = self.true_intensities[:, 0:348]
            true_intensities_b = self.true_intensities[:, 348:]
            pred_intensities_a = self.pred_intensities[:, 0:348]
            pred_intensities_b = self.pred_intensities[:, 348:]
            mask_observed_valid_a = FragmentsRatio.get_mask_observed_valid(true_intensities_a)
            mask_observed_valid_b = FragmentsRatio.get_mask_observed_valid(true_intensities_b)
            observed_boolean_a = FragmentsRatio.make_boolean(true_intensities_a, mask_observed_valid_a)
            observed_boolean_b = FragmentsRatio.make_boolean(true_intensities_b, mask_observed_valid_b)
            predicted_boolean_a = FragmentsRatio.make_boolean(pred_intensities_a, mask_observed_valid_a, cutoff=0.05)
            predicted_boolean_b = FragmentsRatio.make_boolean(pred_intensities_b, mask_observed_valid_b, cutoff=0.05)
            observation_state_a = FragmentsRatio.get_observation_state(
                observed_boolean_a, predicted_boolean_a, mask_observed_valid_a
            )
            observation_state_b = FragmentsRatio.get_observation_state(
                observed_boolean_b, predicted_boolean_b, mask_observed_valid_b
            )
            valid_ions_a = np.maximum(1, FragmentsRatio.count_with_ion_mask(mask_observed_valid_a, xl=True))
            valid_ions_b = np.maximum(1, FragmentsRatio.count_with_ion_mask(mask_observed_valid_b, xl=True))
            valid_ions_b_a = np.maximum(
                1, FragmentsRatio.count_with_ion_mask(mask_observed_valid_a, constants.B_ION_MASK_XL, xl=True)
            )
            valid_ions_b_b = np.maximum(
                1, FragmentsRatio.count_with_ion_mask(mask_observed_valid_b, constants.B_ION_MASK_XL, xl=True)
            )
            valid_ions_y_a = np.maximum(
                1, FragmentsRatio.count_with_ion_mask(mask_observed_valid_a, constants.Y_ION_MASK_XL, xl=True)
            )
            valid_ions_y_b = np.maximum(
                1, FragmentsRatio.count_with_ion_mask(mask_observed_valid_b, constants.Y_ION_MASK_XL, xl=True)
            )
            # counting metrics
            self.metrics_val["count_predicted_a"] = FragmentsRatio.count_with_ion_mask(predicted_boolean_a, xl=True)
            self.metrics_val["count_predicted_b"] = FragmentsRatio.count_with_ion_mask(predicted_boolean_b, xl=True)
            self.metrics_val["count_predicted_b_a"] = FragmentsRatio.count_with_ion_mask(
                predicted_boolean_a, constants.B_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_predicted_b_b"] = FragmentsRatio.count_with_ion_mask(
                predicted_boolean_b, constants.B_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_predicted_y_a"] = FragmentsRatio.count_with_ion_mask(
                predicted_boolean_a, constants.Y_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_predicted_y_b"] = FragmentsRatio.count_with_ion_mask(
                predicted_boolean_b, constants.Y_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_observed_a"] = FragmentsRatio.count_with_ion_mask(observed_boolean_a, xl=True)
            self.metrics_val["count_observed_b"] = FragmentsRatio.count_with_ion_mask(observed_boolean_b, xl=True)
            self.metrics_val["count_observed_b_a"] = FragmentsRatio.count_with_ion_mask(
                observed_boolean_a, constants.B_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_observed_b_b"] = FragmentsRatio.count_with_ion_mask(
                observed_boolean_b, constants.B_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_observed_y_a"] = FragmentsRatio.count_with_ion_mask(
                observed_boolean_a, constants.Y_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_observed_y_b"] = FragmentsRatio.count_with_ion_mask(
                observed_boolean_b, constants.Y_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_observed_and_predicted_a"] = FragmentsRatio.count_observation_states(
                observation_state_a, ObservationState.OBS_AND_PRED, xl=True
            )
            self.metrics_val["count_observed_and_predicted_b"] = FragmentsRatio.count_observation_states(
                observation_state_b, ObservationState.OBS_AND_PRED, xl=True
            )
            self.metrics_val["count_observed_and_predicted_b_a"] = FragmentsRatio.count_observation_states(
                observation_state_a, ObservationState.OBS_AND_PRED, constants.B_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_observed_and_predicted_b_b"] = FragmentsRatio.count_observation_states(
                observation_state_b, ObservationState.OBS_AND_PRED, constants.B_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_observed_and_predicted_y_a"] = FragmentsRatio.count_observation_states(
                observation_state_a, ObservationState.OBS_AND_PRED, constants.Y_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_observed_and_predicted_y_b"] = FragmentsRatio.count_observation_states(
                observation_state_b, ObservationState.OBS_AND_PRED, constants.Y_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_not_observed_and_not_predicted_a"] = FragmentsRatio.count_observation_states(
                observation_state_a, ObservationState.NOT_OBS_AND_NOT_PRED, xl=True
            )
            self.metrics_val["count_not_observed_and_not_predicted_b"] = FragmentsRatio.count_observation_states(
                observation_state_b, ObservationState.NOT_OBS_AND_NOT_PRED, xl=True
            )
            self.metrics_val["count_not_observed_and_not_predicted_b_a"] = FragmentsRatio.count_observation_states(
                observation_state_a, ObservationState.NOT_OBS_AND_NOT_PRED, constants.B_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_not_observed_and_not_predicted_b_b"] = FragmentsRatio.count_observation_states(
                observation_state_b, ObservationState.NOT_OBS_AND_NOT_PRED, constants.B_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_not_observed_and_not_predicted_y_a"] = FragmentsRatio.count_observation_states(
                observation_state_a, ObservationState.NOT_OBS_AND_NOT_PRED, constants.Y_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_not_observed_and_not_predicted_y_b"] = FragmentsRatio.count_observation_states(
                observation_state_b, ObservationState.NOT_OBS_AND_NOT_PRED, constants.Y_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_observed_but_not_predicted_a"] = FragmentsRatio.count_observation_states(
                observation_state_a, ObservationState.OBS_BUT_NOT_PRED, xl=True
            )
            self.metrics_val["count_observed_but_not_predicted_b"] = FragmentsRatio.count_observation_states(
                observation_state_b, ObservationState.OBS_BUT_NOT_PRED, xl=True
            )
            self.metrics_val["count_observed_but_not_predicted_b_a"] = FragmentsRatio.count_observation_states(
                observation_state_a, ObservationState.OBS_BUT_NOT_PRED, constants.B_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_observed_but_not_predicted_b_b"] = FragmentsRatio.count_observation_states(
                observation_state_b, ObservationState.OBS_BUT_NOT_PRED, constants.B_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_observed_but_not_predicted_y_a"] = FragmentsRatio.count_observation_states(
                observation_state_a, ObservationState.OBS_BUT_NOT_PRED, constants.Y_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_observed_but_not_predicted_y_b"] = FragmentsRatio.count_observation_states(
                observation_state_b, ObservationState.OBS_BUT_NOT_PRED, constants.Y_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_not_observed_but_predicted_a"] = FragmentsRatio.count_observation_states(
                observation_state_a, ObservationState.NOT_OBS_BUT_PRED, xl=True
            )
            self.metrics_val["count_not_observed_but_predicted_b"] = FragmentsRatio.count_observation_states(
                observation_state_b, ObservationState.NOT_OBS_BUT_PRED, xl=True
            )
            self.metrics_val["count_not_observed_but_predicted_b_a"] = FragmentsRatio.count_observation_states(
                observation_state_a, ObservationState.NOT_OBS_BUT_PRED, constants.B_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_not_observed_but_predicted_b_b"] = FragmentsRatio.count_observation_states(
                observation_state_b, ObservationState.NOT_OBS_BUT_PRED, constants.B_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_not_observed_but_predicted_y_a"] = FragmentsRatio.count_observation_states(
                observation_state_a, ObservationState.NOT_OBS_BUT_PRED, constants.Y_ION_MASK_XL, xl=True
            )
            self.metrics_val["count_not_observed_but_predicted_y_b"] = FragmentsRatio.count_observation_states(
                observation_state_b, ObservationState.NOT_OBS_BUT_PRED, constants.Y_ION_MASK_XL, xl=True
            )
            # fractional count metrics
            self.metrics_val["fraction_predicted_a"] = self.metrics_val["count_predicted_a"].values / valid_ions_a
            self.metrics_val["fraction_predicted_b"] = self.metrics_val["count_predicted_b"].values / valid_ions_b
            self.metrics_val["fraction_predicted_b_a"] = self.metrics_val["count_predicted_b_a"] / valid_ions_b_a
            self.metrics_val["fraction_predicted_b_b"] = self.metrics_val["count_predicted_b_b"] / valid_ions_b_b
            self.metrics_val["fraction_predicted_y_a"] = self.metrics_val["count_predicted_y_a"] / valid_ions_y_a
            self.metrics_val["fraction_predicted_y_b"] = self.metrics_val["count_predicted_y_b"] / valid_ions_y_b
            self.metrics_val["fraction_observed_a"] = self.metrics_val["count_observed_a"] / valid_ions_a
            self.metrics_val["fraction_observed_b"] = self.metrics_val["count_observed_b"] / valid_ions_b
            self.metrics_val["fraction_observed_b_a"] = self.metrics_val["count_observed_b_a"] / valid_ions_b_a
            self.metrics_val["fraction_observed_b_b"] = self.metrics_val["count_observed_b_b"] / valid_ions_b_b
            self.metrics_val["fraction_observed_y_a"] = self.metrics_val["count_observed_y_a"] / valid_ions_y_a
            self.metrics_val["fraction_observed_y_b"] = self.metrics_val["count_observed_y_b"] / valid_ions_y_b
            self.metrics_val["fraction_observed_and_predicted_a"] = (
                self.metrics_val["count_observed_and_predicted_a"] / valid_ions_a
            )
            self.metrics_val["fraction_observed_and_predicted_b"] = (
                self.metrics_val["count_observed_and_predicted_b"] / valid_ions_b
            )
            self.metrics_val["fraction_observed_and_predicted_b_a"] = (
                self.metrics_val["count_observed_and_predicted_b_a"] / valid_ions_b_a
            )
            self.metrics_val["fraction_observed_and_predicted_b_b"] = (
                self.metrics_val["count_observed_and_predicted_b_b"] / valid_ions_b_b
            )
            self.metrics_val["fraction_observed_and_predicted_y_a"] = (
                self.metrics_val["count_observed_and_predicted_y_a"] / valid_ions_y_a
            )
            self.metrics_val["fraction_observed_and_predicted_y_b"] = (
                self.metrics_val["count_observed_and_predicted_y_b"] / valid_ions_y_b
            )
            self.metrics_val["fraction_not_observed_and_not_predicted_a"] = (
                self.metrics_val["count_not_observed_and_not_predicted_a"] / valid_ions_a
            )
            self.metrics_val["fraction_not_observed_and_not_predicted_b"] = (
                self.metrics_val["count_not_observed_and_not_predicted_b"] / valid_ions_b
            )
            self.metrics_val["fraction_not_observed_and_not_predicted_b_a"] = (
                self.metrics_val["count_not_observed_and_not_predicted_b_a"] / valid_ions_b_a
            )
            self.metrics_val["fraction_not_observed_and_not_predicted_b_b"] = (
                self.metrics_val["count_not_observed_and_not_predicted_b_b"] / valid_ions_b_b
            )
            self.metrics_val["fraction_not_observed_and_not_predicted_y_a"] = (
                self.metrics_val["count_not_observed_and_not_predicted_y_a"] / valid_ions_y_a
            )
            self.metrics_val["fraction_not_observed_and_not_predicted_y_b"] = (
                self.metrics_val["count_not_observed_and_not_predicted_y_b"] / valid_ions_y_b
            )
            self.metrics_val["fraction_observed_but_not_predicted_a"] = (
                self.metrics_val["count_observed_but_not_predicted_a"] / valid_ions_a
            )
            self.metrics_val["fraction_observed_but_not_predicted_b"] = (
                self.metrics_val["count_observed_but_not_predicted_b"] / valid_ions_b
            )
            self.metrics_val["fraction_observed_but_not_predicted_b_a"] = (
                self.metrics_val["count_observed_but_not_predicted_b_a"] / valid_ions_b_a
            )
            self.metrics_val["fraction_observed_but_not_predicted_b_b"] = (
                self.metrics_val["count_observed_but_not_predicted_b_b"] / valid_ions_b_b
            )
            self.metrics_val["fraction_observed_but_not_predicted_y_a"] = (
                self.metrics_val["count_observed_but_not_predicted_y_a"] / valid_ions_y_a
            )
            self.metrics_val["fraction_observed_but_not_predicted_y_b"] = (
                self.metrics_val["count_observed_but_not_predicted_y_b"] / valid_ions_y_b
            )
            self.metrics_val["fraction_not_observed_but_predicted_a"] = (
                self.metrics_val["count_not_observed_but_predicted_a"] / valid_ions_a
            )
            self.metrics_val["fraction_not_observed_but_predicted_b"] = (
                self.metrics_val["count_not_observed_but_predicted_b"] / valid_ions_b
            )
            self.metrics_val["fraction_not_observed_but_predicted_b_a"] = (
                self.metrics_val["count_not_observed_but_predicted_b_a"] / valid_ions_b_a
            )
            self.metrics_val["fraction_not_observed_but_predicted_b_b"] = (
                self.metrics_val["count_not_observed_but_predicted_b_b"] / valid_ions_b_b
            )
            self.metrics_val["fraction_not_observed_but_predicted_y_a"] = (
                self.metrics_val["count_not_observed_but_predicted_y_a"] / valid_ions_y_a
            )
            self.metrics_val["fraction_not_observed_but_predicted_y_b"] = (
                self.metrics_val["count_not_observed_but_predicted_y_b"] / valid_ions_y_b
            )
            # fractional count metrics relative to predictions
            num_predicted_ions_a = np.maximum(1, self.metrics_val["count_predicted_a"])
            num_predicted_ions_b = np.maximum(1, self.metrics_val["count_predicted_b"])
            num_predicted_ions_b_a = np.maximum(1, self.metrics_val["count_predicted_b_a"])
            num_predicted_ions_b_b = np.maximum(1, self.metrics_val["count_predicted_b_b"])
            num_predicted_ions_y_a = np.maximum(1, self.metrics_val["count_predicted_y_a"])
            num_predicted_ions_y_b = np.maximum(1, self.metrics_val["count_predicted_y_b"])
            self.metrics_val["fraction_observed_and_predicted_vs_predicted_a"] = (
                self.metrics_val["count_observed_and_predicted_a"] / num_predicted_ions_a
            )
            self.metrics_val["fraction_observed_and_predicted_vs_predicted_b"] = (
                self.metrics_val["count_observed_and_predicted_b"] / num_predicted_ions_b
            )
            self.metrics_val["fraction_observed_and_predicted_b_vs_predicted_b_a"] = (
                self.metrics_val["count_observed_and_predicted_b_a"] / num_predicted_ions_b_a
            )
            self.metrics_val["fraction_observed_and_predicted_b_vs_predicted_b_b"] = (
                self.metrics_val["count_observed_and_predicted_b_b"] / num_predicted_ions_b_b
            )
            self.metrics_val["fraction_observed_and_predicted_y_vs_predicted_y_a"] = (
                self.metrics_val["count_observed_and_predicted_y_a"] / num_predicted_ions_y_a
            )
            self.metrics_val["fraction_observed_and_predicted_y_vs_predicted_y_b"] = (
                self.metrics_val["count_observed_and_predicted_y_b"] / num_predicted_ions_y_b
            )
            self.metrics_val["fraction_not_observed_and_not_predicted_vs_predicted_a"] = (
                self.metrics_val["count_not_observed_and_not_predicted_a"] / num_predicted_ions_a
            )
            self.metrics_val["fraction_not_observed_and_not_predicted_vs_predicted_b"] = (
                self.metrics_val["count_not_observed_and_not_predicted_b"] / num_predicted_ions_b
            )
            self.metrics_val["fraction_not_observed_and_not_predicted_b_vs_predicted_b_a"] = (
                self.metrics_val["count_not_observed_and_not_predicted_b_a"] / num_predicted_ions_b_a
            )
            self.metrics_val["fraction_not_observed_and_not_predicted_b_vs_predicted_b_b"] = (
                self.metrics_val["count_not_observed_and_not_predicted_b_b"] / num_predicted_ions_b_b
            )
            self.metrics_val["fraction_not_observed_and_not_predicted_y_vs_predicted_y_a"] = (
                self.metrics_val["count_not_observed_and_not_predicted_y_a"] / num_predicted_ions_y_a
            )
            self.metrics_val["fraction_not_observed_and_not_predicted_y_vs_predicted_y_b"] = (
                self.metrics_val["count_not_observed_and_not_predicted_y_b"] / num_predicted_ions_y_b
            )
            self.metrics_val["fraction_observed_but_not_predicted_vs_predicted_a"] = (
                self.metrics_val["count_observed_but_not_predicted_a"] / num_predicted_ions_a
            )
            self.metrics_val["fraction_observed_but_not_predicted_vs_predicted_b"] = (
                self.metrics_val["count_observed_but_not_predicted_b"] / num_predicted_ions_b
            )
            self.metrics_val["fraction_observed_but_not_predicted_b_vs_predicted_b_a"] = (
                self.metrics_val["count_observed_but_not_predicted_b_a"] / num_predicted_ions_b_a
            )
            self.metrics_val["fraction_observed_but_not_predicted_b_vs_predicted_b_b"] = (
                self.metrics_val["count_observed_but_not_predicted_b_b"] / num_predicted_ions_b_b
            )
            self.metrics_val["fraction_observed_but_not_predicted_y_vs_predicted_y_a"] = (
                self.metrics_val["count_observed_but_not_predicted_y_a"] / num_predicted_ions_y_a
            )
            self.metrics_val["fraction_observed_but_not_predicted_y_vs_predicted_y_b"] = (
                self.metrics_val["count_observed_but_not_predicted_y_b"] / num_predicted_ions_y_b
            )

        else:
            mask_observed_valid = FragmentsRatio.get_mask_observed_valid(self.true_intensities)
            observed_boolean = FragmentsRatio.make_boolean(self.true_intensities, mask_observed_valid)
            predicted_boolean = FragmentsRatio.make_boolean(self.pred_intensities, mask_observed_valid, cutoff=0.05)
            observation_state = FragmentsRatio.get_observation_state(
                observed_boolean, predicted_boolean, mask_observed_valid
            )
            valid_ions = np.maximum(1, FragmentsRatio.count_with_ion_mask(mask_observed_valid))
            valid_ions_b = np.maximum(1, FragmentsRatio.count_with_ion_mask(mask_observed_valid, constants.B_ION_MASK))
            valid_ions_y = np.maximum(1, FragmentsRatio.count_with_ion_mask(mask_observed_valid, constants.Y_ION_MASK))

            # counting metrics
            self.metrics_val["count_predicted"] = FragmentsRatio.count_with_ion_mask(predicted_boolean)
            self.metrics_val["count_predicted_b"] = FragmentsRatio.count_with_ion_mask(
                predicted_boolean, constants.B_ION_MASK
            )
            self.metrics_val["count_predicted_y"] = FragmentsRatio.count_with_ion_mask(
                predicted_boolean, constants.Y_ION_MASK
            )

            self.metrics_val["count_observed"] = FragmentsRatio.count_with_ion_mask(observed_boolean)
            self.metrics_val["count_observed_b"] = FragmentsRatio.count_with_ion_mask(
                observed_boolean, constants.B_ION_MASK
            )
            self.metrics_val["count_observed_y"] = FragmentsRatio.count_with_ion_mask(
                observed_boolean, constants.Y_ION_MASK
            )

            self.metrics_val["count_observed_and_predicted"] = FragmentsRatio.count_observation_states(
                observation_state, ObservationState.OBS_AND_PRED
            )
            self.metrics_val["count_observed_and_predicted_b"] = FragmentsRatio.count_observation_states(
                observation_state, ObservationState.OBS_AND_PRED, constants.B_ION_MASK
            )
            self.metrics_val["count_observed_and_predicted_y"] = FragmentsRatio.count_observation_states(
                observation_state, ObservationState.OBS_AND_PRED, constants.Y_ION_MASK
            )

            self.metrics_val["count_not_observed_and_not_predicted"] = FragmentsRatio.count_observation_states(
                observation_state, ObservationState.NOT_OBS_AND_NOT_PRED
            )
            self.metrics_val["count_not_observed_and_not_predicted_b"] = FragmentsRatio.count_observation_states(
                observation_state, ObservationState.NOT_OBS_AND_NOT_PRED, constants.B_ION_MASK
            )
            self.metrics_val["count_not_observed_and_not_predicted_y"] = FragmentsRatio.count_observation_states(
                observation_state, ObservationState.NOT_OBS_AND_NOT_PRED, constants.Y_ION_MASK
            )

            self.metrics_val["count_observed_but_not_predicted"] = FragmentsRatio.count_observation_states(
                observation_state, ObservationState.OBS_BUT_NOT_PRED
            )
            self.metrics_val["count_observed_but_not_predicted_b"] = FragmentsRatio.count_observation_states(
                observation_state, ObservationState.OBS_BUT_NOT_PRED, constants.B_ION_MASK
            )
            self.metrics_val["count_observed_but_not_predicted_y"] = FragmentsRatio.count_observation_states(
                observation_state, ObservationState.OBS_BUT_NOT_PRED, constants.Y_ION_MASK
            )

            self.metrics_val["count_not_observed_but_predicted"] = FragmentsRatio.count_observation_states(
                observation_state, ObservationState.NOT_OBS_BUT_PRED
            )
            self.metrics_val["count_not_observed_but_predicted_b"] = FragmentsRatio.count_observation_states(
                observation_state, ObservationState.NOT_OBS_BUT_PRED, constants.B_ION_MASK
            )
            self.metrics_val["count_not_observed_but_predicted_y"] = FragmentsRatio.count_observation_states(
                observation_state, ObservationState.NOT_OBS_BUT_PRED, constants.Y_ION_MASK
            )

            # fractional count metrics
            self.metrics_val["fraction_predicted"] = self.metrics_val["count_predicted"].values / valid_ions
            self.metrics_val["fraction_predicted_b"] = self.metrics_val["count_predicted_b"] / valid_ions_b
            self.metrics_val["fraction_predicted_y"] = self.metrics_val["count_predicted_y"] / valid_ions_y

            self.metrics_val["fraction_observed"] = self.metrics_val["count_observed"] / valid_ions
            self.metrics_val["fraction_observed_b"] = self.metrics_val["count_observed_b"] / valid_ions_b
            self.metrics_val["fraction_observed_y"] = self.metrics_val["count_observed_y"] / valid_ions_y

            self.metrics_val["fraction_observed_and_predicted"] = (
                self.metrics_val["count_observed_and_predicted"] / valid_ions
            )
            self.metrics_val["fraction_observed_and_predicted_b"] = (
                self.metrics_val["count_observed_and_predicted_b"] / valid_ions_b
            )
            self.metrics_val["fraction_observed_and_predicted_y"] = (
                self.metrics_val["count_observed_and_predicted_y"] / valid_ions_y
            )

            self.metrics_val["fraction_not_observed_and_not_predicted"] = (
                self.metrics_val["count_not_observed_and_not_predicted"] / valid_ions
            )
            self.metrics_val["fraction_not_observed_and_not_predicted_b"] = (
                self.metrics_val["count_not_observed_and_not_predicted_b"] / valid_ions_b
            )
            self.metrics_val["fraction_not_observed_and_not_predicted_y"] = (
                self.metrics_val["count_not_observed_and_not_predicted_y"] / valid_ions_y
            )

            self.metrics_val["fraction_observed_but_not_predicted"] = (
                self.metrics_val["count_observed_but_not_predicted"] / valid_ions
            )
            self.metrics_val["fraction_observed_but_not_predicted_b"] = (
                self.metrics_val["count_observed_but_not_predicted_b"] / valid_ions_b
            )
            self.metrics_val["fraction_observed_but_not_predicted_y"] = (
                self.metrics_val["count_observed_but_not_predicted_y"] / valid_ions_y
            )

            self.metrics_val["fraction_not_observed_but_predicted"] = (
                self.metrics_val["count_not_observed_but_predicted"] / valid_ions
            )
            self.metrics_val["fraction_not_observed_but_predicted_b"] = (
                self.metrics_val["count_not_observed_but_predicted_b"] / valid_ions_b
            )
            self.metrics_val["fraction_not_observed_but_predicted_y"] = (
                self.metrics_val["count_not_observed_but_predicted_y"] / valid_ions_y
            )

            # fractional count metrics relative to predictions
            num_predicted_ions = np.maximum(1, self.metrics_val["count_predicted"])
            num_predicted_ions_b = np.maximum(1, self.metrics_val["count_predicted_b"])
            num_predicted_ions_y = np.maximum(1, self.metrics_val["count_predicted_y"])

            self.metrics_val["fraction_observed_and_predicted_vs_predicted"] = (
                self.metrics_val["count_observed_and_predicted"] / num_predicted_ions
            )
            self.metrics_val["fraction_observed_and_predicted_b_vs_predicted_b"] = (
                self.metrics_val["count_observed_and_predicted_b"] / num_predicted_ions_b
            )
            self.metrics_val["fraction_observed_and_predicted_y_vs_predicted_y"] = (
                self.metrics_val["count_observed_and_predicted_y"] / num_predicted_ions_y
            )

            self.metrics_val["fraction_not_observed_and_not_predicted_vs_predicted"] = (
                self.metrics_val["count_not_observed_and_not_predicted"] / num_predicted_ions
            )
            self.metrics_val["fraction_not_observed_and_not_predicted_b_vs_predicted_b"] = (
                self.metrics_val["count_not_observed_and_not_predicted_b"] / num_predicted_ions_b
            )
            self.metrics_val["fraction_not_observed_and_not_predicted_y_vs_predicted_y"] = (
                self.metrics_val["count_not_observed_and_not_predicted_y"] / num_predicted_ions_y
            )

            self.metrics_val["fraction_observed_but_not_predicted_vs_predicted"] = (
                self.metrics_val["count_observed_but_not_predicted"] / num_predicted_ions
            )
            self.metrics_val["fraction_observed_but_not_predicted_b_vs_predicted_b"] = (
                self.metrics_val["count_observed_but_not_predicted_b"] / num_predicted_ions_b
            )
            self.metrics_val["fraction_observed_but_not_predicted_y_vs_predicted_y"] = (
                self.metrics_val["count_observed_but_not_predicted_y"] / num_predicted_ions_y
            )

            # not needed, as these are simply (1 - fraction_observed_and_predicted_vs_predicted)
            self.metrics_val["fraction_not_observed_but_predicted_vs_predicted"] = (
                self.metrics_val["count_not_observed_but_predicted"] / num_predicted_ions
            )
            self.metrics_val["fraction_not_observed_but_predicted_b_vs_predicted"] = (
                self.metrics_val["count_not_observed_but_predicted_b"] / num_predicted_ions_b
            )
            self.metrics_val["fraction_not_observed_but_predicted_y_vs_predicted"] = (
                self.metrics_val["count_not_observed_but_predicted_y"] / num_predicted_ions_y
            )
