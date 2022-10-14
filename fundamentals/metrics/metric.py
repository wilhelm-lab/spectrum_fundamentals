from abc import abstractmethod

import numpy as np
import pandas as pd


class Metric:
    """Main to init a Metric obj."""

    # check https://gitlab.lrz.de/proteomics/prosit_tools/oktoberfest/-/blob/develop/oktoberfest/rescoring/annotate.R
    # for all metrics
    pred_intensities: np.ndarray  # list of lists
    true_intensities: np.ndarray  # list of lists
    metrics_val: pd.DataFrame

    def __init__(self, pred_intensities: np.ndarray, true_intensities: np.ndarray):
        """
        Initialize a Metric object.

        :param pred_intensities: predicted intensities
        :param true_intensities: observed intensities
        """
        self.pred_intensities = pred_intensities
        self.true_intensities = true_intensities
        self.metrics_val = pd.DataFrame()

    @abstractmethod
    def calc(self, all_features: bool):
        """Calculate."""
        pass

    def write_to_file(self, file_path: str):
        """Write to file_path."""
        self.metrics_val.to_csv(file_path, sep="\t", index=False)
