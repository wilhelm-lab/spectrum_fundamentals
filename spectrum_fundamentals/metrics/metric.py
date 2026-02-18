from abc import abstractmethod
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse

from spectrum_fundamentals import constants

SEQ_LEN = 30


class Metric:
    """Main to init a Metric obj."""

    # check https://gitlab.lrz.de/proteomics/prosit_tools/oktoberfest/-/blob/develop/oktoberfest/rescoring/annotate.R
    # for all metrics
    pred_intensities: Optional[Union[np.ndarray, scipy.sparse.csr_matrix]]  # list of lists
    true_intensities: Optional[Union[np.ndarray, scipy.sparse.csr_matrix]]  # list of lists
    metrics_val: pd.DataFrame

    def __init__(
        self,
        pred_intensities: Optional[Union[np.ndarray, scipy.sparse.csr_matrix]] = None,
        true_intensities: Optional[Union[np.ndarray, scipy.sparse.csr_matrix]] = None,
        mz: Optional[Union[np.ndarray, scipy.sparse.csr_matrix]] = None,
        xl: bool = False,
        cms2: bool = False,
        task: str = "default",
        featured_ions: Optional[List[str]] = None,
    ):
        """
        Initialize a Metric object.

        :param pred_intensities: predicted intensities
        :param true_intensities: observed intensities
        :param mz: observed mz values
        :param xl: whether the metric is used for crosslinked or linear peptides
        :param cms2: if cross-ling CM
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
    def calc(self, all_features: bool):
        """Calculate."""
        pass

    def write_to_file(self, file_path: str):
        """Write to file_path."""
        self.metrics_val.to_csv(file_path, sep="\t", index=False)
