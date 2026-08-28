from abc import abstractmethod

import numpy as np
import pandas as pd
import scipy.sparse

from spectrum_fundamentals import constants

SEQ_LEN = 30


class Metric:
    """Main to init a Metric obj."""

    # check https://gitlab.lrz.de/proteomics/prosit_tools/oktoberfest/-/blob/develop/oktoberfest/rescoring/annotate.R
    # for all metrics
    pred_intensities: np.ndarray | scipy.sparse.csr_matrix | None  # list of lists
    true_intensities: np.ndarray | scipy.sparse.csr_matrix | None  # list of lists
    metrics_val: pd.DataFrame

    def __init__(
        self,
        pred_intensities: np.ndarray | scipy.sparse.csr_matrix | None = None,
        true_intensities: np.ndarray | scipy.sparse.csr_matrix | None = None,
        mz: np.ndarray | scipy.sparse.csr_matrix | None = None,
        xl: bool = False,
        cms2: bool = False,
        all_features_flag: bool = False,
        sc_features_flag: bool = False,
        task: str = "default",
        featured_ions: list[str] | None = None,
    ):
        """
        Initialize a Metric object.

        :param pred_intensities: predicted intensities
        :param true_intensities: observed intensities
        :param mz: observed mz values
        :param xl: whether the metric is used for crosslinked or linear peptides
        :param cms2: if cross-ling CM
        :param all_features_flag: if True, calculate all metrics
        :param sc_features_flag: if True, calculate the additional single-cell rescoring features
                                 (peak-matching quality, coverage, ion-series continuity, TMT reporter
                                 ions and the b1-excluded spectral angle). Implied by all_features_flag.
        :param task: define which workflows will be used
        :param featured_ions: list of ions will be used to generate features
        """
        self.pred_intensities = pred_intensities
        self.true_intensities = true_intensities
        self.mz = mz
        self.metrics_val = pd.DataFrame()
        self.xl = xl
        self.cms2 = cms2
        self.task = task
        if featured_ions is None:
            featured_ions = ["b", "y"]
        self.featured_ions = featured_ions
        self.max_length = 348 if cms2 else 174
        self.all_features_flag = all_features_flag
        # all_features implies sc_features: "give me everything" must not silently skip a family.
        self.sc_features_flag = sc_features_flag or all_features_flag

        reps = (SEQ_LEN - 1) * (2 if self.cms2 else 1)
        if self.task == "default":
            b = np.tile([0, 0, 0, 1, 1, 1], reps)
            y = np.tile([1, 1, 1, 0, 0, 0], reps)

            self.ion_mask = {"b": b, "y": y}

            self.mask_dict = {
                1: np.tile([1, 0, 0, 1, 0, 0], reps),
                2: np.tile([0, 1, 0, 0, 1, 0], reps),
                3: np.tile([0, 0, 1, 0, 0, 1], reps),
            }

        elif self.task == "multifrag":
            self.ion_mask = {
                ion: (constants.ION_DIC["type"] == ion).to_numpy().astype(int) for ion in self.featured_ions
            }
            self.mask_dict = {
                1: (constants.ION_DIC["charge"] == 1).to_numpy().astype(int),
                2: (constants.ION_DIC["charge"] == 2).to_numpy().astype(int),
                3: (constants.ION_DIC["charge"] == 3).to_numpy().astype(int),
            }

    @abstractmethod
    def calc(self):
        """Calculate."""
        pass

    def b1_mask(self) -> np.ndarray:
        """
        Build a mask that zeroes the singly- and multiply-charged b1 slots of the intensity vector.

        b1 ions are thermodynamically unstable under HCD and are rarely observed, so a PSM should not
        be penalised for their absence. Their position depends on the vector layout, which differs per
        task, so the mask is derived rather than hard-coded:

        * ``task="default"``: :py:data:`constants.ANNOTATION_FRAGMENT_TYPE` / ``_NUMBER``, the same
          arrays the annotation vector is built from; b1 lands on indices 3-5 of each peptide's block
          (twice, at an offset of ``VEC_LENGTH``, when ``cms2``).
        * ``task="multifrag"``: :py:data:`constants.ION_DIC`, sorted by ion name, where b1 sits at a
          completely unrelated index (3-5 there are A-ions).

        :return: mask of shape (1, vector length), 1.0 everywhere except the b1 slots
        """
        if self.task == "multifrag":
            ion_type, fragment_number = constants.ION_DIC["type"], constants.ION_DIC["num"]
        else:
            # same source the annotation vector itself is built from, so the two cannot drift apart
            ion_type = np.asarray(constants.ANNOTATION_FRAGMENT_TYPE)
            fragment_number = np.asarray(constants.ANNOTATION_FRAGMENT_NUMBER)
        is_b1 = np.asarray((ion_type == "b") & (fragment_number == 1))
        if self.task != "multifrag" and self.cms2:
            is_b1 = np.tile(is_b1, 2)  # the vector holds one block per crosslinked peptide
        mask = np.ones((1, is_b1.shape[0]))
        mask[:, is_b1] = 0.0
        return mask

    def write_to_file(self, file_path: str):
        """Write to file_path."""
        self.metrics_val.to_csv(file_path, sep="\t", index=False)
