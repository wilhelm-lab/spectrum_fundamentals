import enum
import logging
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats
from moepy import lowess
from scipy import interpolate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from . import fragments_ratio as fr
from . import similarity as sim
from .metric import Metric

logger = logging.getLogger(__name__)


class TargetDecoyLabel(enum.IntEnum):
    """Target and decoy labels as used by Percolator."""

    TARGET = 1
    DECOY = -1


class Percolator(Metric):
    """
    Expects the following metadata columns.

    RAW_FILE
    SCAN_NUMBER
    MODIFIED_SEQUENCE: sequence with modifications
    SEQUENCE: sequence without modifications
    CHARGE: precursor charge state
    MASS: experimental precursor mass
    CALCULATED_MASS: calculated mass based on sequence and modifications
    SCORE: Andromeda score
    REVERSE: does the sequence come from the reversed (=decoy) database
    FRAGMENTATION: fragmentation method, e.g. HCD, CID
    RETENTION_TIME: observed retention time
    PREDICTED_RETENTION_TIME: predicted retention time by Prosit
    """

    metadata: pd.DataFrame
    target_decoy_labels: np.ndarray
    input_type: str
    fdr_cutoff: float

    def __init__(
        self,
        metadata: pd.DataFrame,
        input_type: str,
        pred_intensities: Optional[Union[np.ndarray, scipy.sparse.csr_matrix]] = None,
        true_intensities: Optional[Union[np.ndarray, scipy.sparse.csr_matrix]] = None,
        mz: Optional[Union[np.ndarray, scipy.sparse.csr_matrix]] = None,
        all_features_flag: bool = False,
        regression_method: str = "lowess",
        fdr_cutoff: float = 0.01,
    ):
        """Initialize a Percolator obj."""
        self.metadata = metadata
        self.input_type = input_type
        self.all_features_flag = all_features_flag
        self.regression_method = regression_method
        self.fdr_cutoff = fdr_cutoff
        super().__init__(pred_intensities, true_intensities, mz)

    @staticmethod
    def sample_balanced_over_bins(retention_time_df: pd.DataFrame, sample_size: int = 5000) -> pd.Index:
        """
        Sample balanced over bins.

        :param retention_time_df: DataFrame with observed and predicted retention times
        :param sample_size: number of samples
        :return: RT Index
        """
        # bin retention times
        # print(retention_time_df['RETENTION_TIME'])
        min_rt = retention_time_df["RETENTION_TIME"].min() * 0.99
        max_rt = retention_time_df["RETENTION_TIME"].max() * 1.01
        bin_width = (
            2
            * scipy.stats.iqr(retention_time_df["RETENTION_TIME"])
            / len(retention_time_df["RETENTION_TIME"]) ** (1 / 3)
        )  # Freedman–Diaconis rule
        break_points = np.arange(min_rt, max_rt, bin_width)
        retention_time_df["rt_bin_index"] = np.digitize(retention_time_df["RETENTION_TIME"], break_points)

        # sample a subset in each bin. Arbitrary target is 5000 datapoints spread over the bin counts
        points_per_bin = int(np.floor(sample_size / len(break_points)))
        retention_time_df = retention_time_df.groupby("rt_bin_index").apply(
            lambda x: pd.DataFrame.sample(x, n=min(points_per_bin, len(x)), replace=False)
        )
        return retention_time_df.reset_index(level=0, drop=True).index

    @staticmethod
    def get_aligned_predicted_retention_times(
        observed_retention_times_fdr_filtered: Union[np.ndarray, pd.Series],
        predicted_retention_times_fdr_filtered: Union[np.ndarray, pd.Series],
        predicted_retention_times_all: Union[np.ndarray, pd.Series],
        curve_fitting_method: str = "lowess",
    ) -> np.ndarray:
        """
        Apply regression to find a mapping from predicted iRT values to experimental retention times.

        :param observed_retention_times_fdr_filtered: observed retention times after FDR filter
        :param predicted_retention_times_fdr_filtered: predicted retention times after FDR filter
        :param predicted_retention_times_all: all predicted retention times
        :param curve_fitting_method: method for curve fitting (lowess, spline, or logistic)
        :return: aligned predicted retention times
        """
        observed_rts = np.array(observed_retention_times_fdr_filtered, dtype=np.float64)
        predicted_rts = np.array(predicted_retention_times_fdr_filtered, dtype=np.float64)

        it = 0  # The number of residual-based reweightings to perform. Don't use the iterative reweighting (it > 1),
        # this result in NaNs

        # TODO: use Akaike information criterion to choose a good value for frac
        frac = 0.5  # Between 0 and 1. The fraction of the data used when estimating each y-value.

        fit_func = get_fitting_func(curve_fitting_method)
        discard_percentage = 0.1  # in percents, so 0.1 = 0.1% (not 10%!)
        median_abs_error = 1.0

        while discard_percentage < 50.0 and median_abs_error > 0.02:
            params = fit_func(predicted_rts, observed_rts)
            aligned_rts_predicted = params[0]

            abs_errors = np.abs(aligned_rts_predicted - observed_rts)
            cut_off = np.percentile(abs_errors, 100 - discard_percentage)
            median_abs_error = np.median(np.abs(abs_errors))

            logger.debug(f"Median absolute error aligned rts: {median_abs_error}")

            if median_abs_error > 0.02:
                keep_idxs = np.nonzero(abs_errors < cut_off)
                observed_rts = observed_rts[keep_idxs[0]]
                predicted_rts = predicted_rts[keep_idxs[0]]

                discard_percentage *= 1.5

        logger.debug(f"Observed RT anchor points:\n{observed_retention_times_fdr_filtered}")
        logger.debug(f"Predicted RT anchor points:\n{predicted_retention_times_fdr_filtered}")

        if curve_fitting_method == "spline":
            aligned_rts_predicted = interpolate.BSpline(*params[1:])(predicted_retention_times_all)
        elif curve_fitting_method == "lowess":
            lowess_model = lowess.Lowess()
            lowess_model.fit(predicted_rts, observed_rts, frac=frac, robust_iters=it)
            aligned_rts_predicted = lowess_model.predict(np.array(predicted_retention_times_all))
        else:  # logistic
            aligned_rts_predicted = logistic(
                predicted_retention_times_all, *opt.curve_fit(logistic, predicted_rts, observed_rts, method="lm")[0]
            )

        return aligned_rts_predicted

    @staticmethod
    def get_delta_score(scores_df: pd.DataFrame, scoring_feature: str) -> np.ndarray:
        """
        Calculates delta scores by sorting (from high to low) and grouping PSMs by scan number.

        Inside each group the delta scores are calculated per PSM to the next best of that group.
        The lowest scoring PSM of each group receives a delta score of 0.
        :param scores_df: must contain two columns: scoring_feature (eg. 'spectral_angle') and 'ScanNr'
        :param scoring_feature: feature name to get the delta scores of
        :raises NotImplementedError: If there is only one unique value for ScanNr in the scores_df.
        :return: numpy array of delta scores
        """
        # TODO: sort after grouping for better efficiency
        scores_df = scores_df.sort_values(by=scoring_feature, ascending=True)
        groups = scores_df.groupby(["ScanNr"])
        t = groups.apply(lambda scores_df_: scores_df_[scoring_feature] - scores_df_[scoring_feature].shift(1))
        # apply doesnt work for one group only
        if len(groups) == 1:
            raise NotImplementedError
        scores_df["delta_" + scoring_feature] = pd.Series(t.reset_index(level=0, drop=True))
        scores_df.fillna(0, inplace=True)
        scores_df.sort_index(inplace=True)
        return scores_df["delta_" + scoring_feature].to_numpy()

    @staticmethod
    def get_specid(metadata_subset: Union[pd.Series, Tuple]) -> str:
        """
        Create a unique identifier used as spectrum id in percolator, this is not parsed by percolator but functions \
        as a key to map percolator results back to our internal representation.

        :param metadata_subset: tuple of (raw_file, scan_number, modified_sequence, charge and optionally scan_event_number)
        :return: percolator spectrum id
        """
        return "-".join([f"{elem}" for elem in metadata_subset])

    @staticmethod
    def count_missed_cleavages(sequence: str) -> int:
        """
        Count number of missed cleavages assuming Trypsin/P proteolysis.

        :param sequence: peptide sequence
        :return: number of missed cleavages
        """
        return sequence[:-1].count("K") + sequence[:-1].count("R")

    @staticmethod
    def count_arginines_and_lysines(sequence: str) -> int:
        """
        Count number of arginines and lysines.

        :param sequence: peptide sequence
        :return: number of arginines and lysines
        """
        return sequence.count("K") + sequence.count("R")

    @staticmethod
    def calculate_mass_difference(metadata_subset: Tuple[float, float]) -> float:
        """
        Calculate mass difference.

        :param metadata_subset: experimental and calculated mass as tuple
        :return: mass difference
        """
        experimental_mass, calculated_mass = metadata_subset
        return calculated_mass - experimental_mass

    @staticmethod
    def calculate_mass_difference_ppm(metadata_subset: Tuple[float, float]) -> float:
        """
        Calculate mass difference in ppm.

        :param metadata_subset: experimental and calculated mass as tuple
        :return: mass difference in ppm
        """
        experimental_mass, calculated_mass = metadata_subset
        return (calculated_mass - experimental_mass) / experimental_mass * 1e6

    @staticmethod
    def get_target_decoy_label(reverse: bool):
        """
        Get target or decoy label.

        :param reverse: if true, return the label for DECOY, otherwise return the label for TARGET
        :return: target/decoy label for percolator
        """
        return TargetDecoyLabel.DECOY if reverse else TargetDecoyLabel.TARGET

    def add_common_features(self):
        """Add features used by both Andromeda and Prosit feature scoring sets."""
        self.metrics_val["missedCleavages"] = self.metadata["SEQUENCE"].apply(Percolator.count_missed_cleavages)
        self.metrics_val["KR"] = self.metadata["SEQUENCE"].apply(Percolator.count_arginines_and_lysines)
        self.metrics_val["sequence_length"] = self.metadata["SEQUENCE"].apply(lambda x: len(x))

        self.metrics_val["Mass"] = self.metadata["CALCULATED_MASS"]  # this is the calculated mass used as a feature
        self.metrics_val["Charge1"] = (self.metadata["PRECURSOR_CHARGE"] == 1).astype(int)
        self.metrics_val["Charge2"] = (self.metadata["PRECURSOR_CHARGE"] == 2).astype(int)
        self.metrics_val["Charge3"] = (self.metadata["PRECURSOR_CHARGE"] == 3).astype(int)
        self.metrics_val["Charge4"] = (self.metadata["PRECURSOR_CHARGE"] == 4).astype(int)
        self.metrics_val["Charge5"] = (self.metadata["PRECURSOR_CHARGE"] == 5).astype(int)
        self.metrics_val["Charge6"] = (self.metadata["PRECURSOR_CHARGE"] == 6).astype(int)

        self.metrics_val["UnknownFragmentationMethod"] = (~self.metadata["FRAGMENTATION"].isin(["HCD", "CID"])).astype(
            int
        )
        self.metrics_val["HCD"] = (self.metadata["FRAGMENTATION"] == "HCD").astype(int)
        self.metrics_val["CID"] = (self.metadata["FRAGMENTATION"] == "CID").astype(int)

    def add_percolator_metadata_columns(self):
        """Add metadata columns needed by percolator, e.g. to identify a PSM."""
        spec_id_cols = ["RAW_FILE", "SCAN_NUMBER", "MODIFIED_SEQUENCE", "PRECURSOR_CHARGE"]
        if "SCAN_EVENT_NUMBER" in self.metadata.columns:
            spec_id_cols.append("SCAN_EVENT_NUMBER")
        self.metrics_val["SpecId"] = self.metadata[spec_id_cols].apply(Percolator.get_specid, axis=1)
        self.metrics_val["Label"] = self.target_decoy_labels
        self.metrics_val["ScanNr"] = self.metadata["SCAN_NUMBER"]
        self.metrics_val["filename"] = self.metadata["RAW_FILE"]
        self.metrics_val["Peptide"] = self.metadata["MODIFIED_SEQUENCE"].apply(lambda x: "_." + x + "._")

        self.metrics_val["Proteins"] = self.metadata[
            "MODIFIED_SEQUENCE"
        ]  # we don't need the protein ID to get PSM / peptide results, fill with peptide sequence

    def apply_lda_and_get_indices_below_fdr(
        self, initial_scoring_feature: str = "spectral_angle", fdr_cutoff: float = 0.01
    ):
        """
        Applies a linear discriminant analysis on the features calculated so far (before retention time alignment) \
        to estimate false discovery rates (FDRs).

        :param initial_scoring_feature: name of the initial scoring feature
        :param fdr_cutoff: FDR cutoff as float
        :return: array with indices below FDR
        """
        target_idxs_below_fdr = self.get_indices_below_fdr(initial_scoring_feature, fdr_cutoff=fdr_cutoff)
        decoy_idxs = np.argwhere(self.target_decoy_labels == TargetDecoyLabel.DECOY).flatten()
        logger.info(
            f"Found {len(target_idxs_below_fdr)} targets and {len(decoy_idxs)} decoys as input for the LDA model"
        )

        lda_idxs = np.concatenate((target_idxs_below_fdr, decoy_idxs)).astype(int)

        x = self.metrics_val.iloc[lda_idxs, :].to_numpy()
        y = self.target_decoy_labels[lda_idxs]

        lda = LinearDiscriminantAnalysis()
        lda.fit(x, y)

        self.metrics_val["lda_scores"] = lda.decision_function(self.metrics_val.to_numpy())

        return self.get_indices_below_fdr("lda_scores", fdr_cutoff=fdr_cutoff)

    def get_indices_below_fdr(self, feature_name: str, fdr_cutoff: float = 0.01) -> np.ndarray:
        """
        Get indices below FDR.

        :param feature_name: name of the feature to sort by as string
        :param fdr_cutoff: FDR cutoff as float
        :return: array with indices below FDR
        """
        scores_df = self.metrics_val[[feature_name]].copy()
        scores_df["Label"] = self.target_decoy_labels
        # scores_df['Sequence'] = self.metadata['SEQUENCE']
        scores_df = scores_df.sort_values(feature_name, ascending=False)
        logger.debug(scores_df.head(100))

        scores_df["fdr"] = Percolator.calculate_fdrs(scores_df["Label"])

        # filter for targets only
        scores_df = scores_df[scores_df["Label"] == TargetDecoyLabel.TARGET]

        accepted_indices = scores_df.index[scores_df["fdr"] < fdr_cutoff]
        if len(accepted_indices) == 0:
            logger.error(
                f"Could not find any targets below {fdr_cutoff} out of {len(scores_df.index)} targets in total"
            )
            return np.array([])

        logger.info(
            f"Found {len(accepted_indices)} (out of {len(scores_df.index)}) targets below {fdr_cutoff} \
            FDR using {feature_name} as feature"
        )

        return np.sort(scores_df.index[: len(accepted_indices)])

    @staticmethod
    def calculate_fdrs(sorted_labels: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Calculate FDR.

        :param sorted_labels: array with labels sorted (target, decoy)
        :return: array with calculated FDRs
        """
        if isinstance(sorted_labels, pd.Series):
            sorted_labels = sorted_labels.to_numpy()
        cumulative_decoy_count = np.cumsum(sorted_labels == TargetDecoyLabel.DECOY) + 1
        cumulative_target_count = np.cumsum(sorted_labels == TargetDecoyLabel.TARGET) + 1
        return Percolator.fdrs_to_qvals(cumulative_decoy_count / cumulative_target_count)

    @staticmethod
    def fdrs_to_qvals(fdrs: np.ndarray) -> np.ndarray:
        """
        Converts FDRs to q-values.

        :param fdrs: array with FDRs
        :return: array with qvals
        """
        qvals = np.zeros(len(fdrs), dtype=float)
        if len(fdrs) > 0:
            qvals[len(fdrs) - 1] = fdrs[-1]
            for i in range(len(fdrs) - 2, -1, -1):
                qvals[i] = min(qvals[i + 1], fdrs[i])
        return qvals

    def _reorder_columns_for_percolator(self):
        all_columns = self.metrics_val.columns
        first_columns = ["SpecId", "Label", "ScanNr", "filename"]
        last_columns = ["Peptide", "Proteins"]
        mid_columns = list(set(all_columns) - set(first_columns) - set(last_columns))
        new_columns = first_columns + sorted(mid_columns) + last_columns
        self.metrics_val = self.metrics_val[new_columns]

    def calc(self):
        """Adds percolator metadata and feature columns to metrics_val based on PSM metadata."""
        self.add_common_features()

        self.target_decoy_labels = self.metadata["REVERSE"].apply(Percolator.get_target_decoy_label).to_numpy()

        np.random.seed(1)
        # add Prosit or Andromeda features
        if self.input_type == "rescore":
            fragments_ratio = fr.FragmentsRatio(self.pred_intensities, self.true_intensities)
            fragments_ratio.calc()

            similarity = sim.SimilarityMetrics(self.pred_intensities, self.true_intensities, self.mz)
            similarity.calc(self.all_features_flag)

            self.metrics_val = pd.concat(
                [self.metrics_val, fragments_ratio.metrics_val, similarity.metrics_val], axis=1
            )

            lda_failed = False
            idxs_below_lda_fdr = self.apply_lda_and_get_indices_below_fdr(fdr_cutoff=self.fdr_cutoff)
            current_fdr = self.fdr_cutoff
            while len(idxs_below_lda_fdr) == 0:
                current_fdr += 0.01
                idxs_below_lda_fdr = self.apply_lda_and_get_indices_below_fdr(fdr_cutoff=current_fdr)
                if current_fdr >= 0.1:
                    lda_failed = True
                    break

            if lda_failed:
                sampled_idxs = Percolator.sample_balanced_over_bins(self.metadata[["RETENTION_TIME", "PREDICTED_IRT"]])
            else:
                sampled_idxs = Percolator.sample_balanced_over_bins(
                    self.metadata[["RETENTION_TIME", "PREDICTED_IRT"]].iloc[idxs_below_lda_fdr, :]
                )

            file_sample = self.metadata.iloc[sampled_idxs].sort_values("PREDICTED_IRT")
            aligned_predicted_rts = Percolator.get_aligned_predicted_retention_times(
                file_sample["RETENTION_TIME"],
                file_sample["PREDICTED_IRT"],
                self.metadata["PREDICTED_IRT"],
                self.regression_method,
            )

            self.metrics_val["RT"] = self.metadata["RETENTION_TIME"]
            self.metrics_val["pred_RT"] = self.metadata["PREDICTED_IRT"]
            self.metrics_val["iRT"] = aligned_predicted_rts
            self.metrics_val["collision_energy_aligned"] = self.metadata["COLLISION_ENERGY"] / 100.0
            self.metrics_val["abs_rt_diff"] = np.abs(self.metadata["RETENTION_TIME"] - aligned_predicted_rts)

            median_abs_error_lda_targets = np.median(self.metrics_val["abs_rt_diff"].iloc[idxs_below_lda_fdr])
            logger.info(
                f"Median absolute error predicted vs observed retention time on targets < 1% FDR: {median_abs_error_lda_targets}"
            )
            logger.debug(self.metrics_val[["RT", "pred_RT", "abs_rt_diff", "lda_scores"]].iloc[idxs_below_lda_fdr, :])
        else:
            self.metrics_val["andromeda"] = self.metadata["SCORE"]

        self.add_percolator_metadata_columns()
        if self.input_type == "rescore":
            # TODO: only add this feature if they are not all zero
            # self.metrics_val['spectral_angle_delta_score'] = Percolator.get_delta_score(self.metrics_val[['ScanNr',
            # 'spectral_angle']], 'spectral_angle')
            pass
        else:
            self.metrics_val["andromeda_delta_score"] = Percolator.get_delta_score(
                self.metrics_val[["ScanNr", "andromeda"]], "andromeda"
            )

        self._reorder_columns_for_percolator()


def get_fitting_func(curve_fitting_method: str):
    """
    Retrieve the correct function given a curve fitting method.

    :param curve_fitting_method: method for curve fitting (lowess, spline, or logistic)
    :raises ValueError: if an invalid curve_fitting_method is supplied
    :return: Callable that accepts x and y, i.e. fit_func(x,y) where x are the data points and y
        are the corresponding measures for which the fit should be done.
    """
    if curve_fitting_method == "logistic":
        return lambda x, y: (logistic(x, *opt.curve_fit(logistic, x, y, method="lm")[0]),)
    elif curve_fitting_method == "lowess":
        return lambda x, y: (lowess.lowess_fit_and_predict(x, y, frac=0.5),)
    elif curve_fitting_method == "spline":
        return lambda x, y: spline(2, x, y)
    else:
        raise ValueError("curve_fitting_method should be one of the following strings: lowess, spline, logistic.")


def spline(knots: int, x: np.ndarray, y: np.ndarray):
    """Calculates spline fitting."""
    x_new = np.linspace(0, 1, knots + 2)[1:-1]
    q_knots = np.quantile(x, x_new)
    t, c, k = interpolate.splrep(x, y, t=q_knots, s=2)
    yfit = interpolate.BSpline(t, c, k)(x)
    return yfit, t, c, k


def logistic(x: Union[pd.Series, np.ndarray], a: float, b: float, c: float, d: float):
    """Calculates logistic regression function."""
    exponent = np.clip(-c * (x - d), -700, 700)  # make this stable, i.e. avoid 0.0 or inf
    return a / (1.0 + np.exp(exponent)) + b
