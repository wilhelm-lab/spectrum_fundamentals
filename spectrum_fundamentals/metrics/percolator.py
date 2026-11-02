import enum
import logging
import math
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats
from moepy import lowess
from scipy import interpolate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from spectrum_fundamentals import constants
from spectrum_fundamentals.metrics import fragments_ratio as fr
from spectrum_fundamentals.metrics import similarity as sim
from spectrum_fundamentals.metrics.metric import Metric
from spectrum_fundamentals.mod_string import internal_without_mods_keep_loc

logger = logging.getLogger(__name__)

# fixed modifications and labels that do not make a PSM modified: carbamidomethyl, iTRAQ 4plex, SILAC heavy K and R,
# iTRAQ 8plex, TMT and TMTpro
STANDARD_UNIMOD_IDS = (4, 214, 259, 267, 730, 737, 2016)


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

    BASE_COLUMNS = [
        "raw_file",
        "scan_number",
        "modified_sequence",
        "precursor_charge",
        "scan_event_number",
        "mass",
        "score",
        "reverse",
        "sequence",
        "peptide_length",
        "fragmentation",
        "calculated_mass",
        "sequence_a",
        "sequence_b",
        "modified_sequence_a",
        "modified_sequence_b",
        "retention_time",
        "predicted_irt",
        "instrument_types",
        "mass_analyzer",
        "mz_range",
        "collision_energy",
        "proteins",
    ]

    def __init__(
        self,
        metadata: pd.DataFrame,
        input_type: str,
        pred_intensities: np.ndarray | scipy.sparse.csr_matrix | None = None,
        true_intensities: np.ndarray | scipy.sparse.csr_matrix | None = None,
        mz: np.ndarray | scipy.sparse.csr_matrix | None = None,
        *,
        all_features_flag: bool = False,
        regression_method: str = "lowess",
        fdr_cutoff: float = 0.01,
        additional_columns: str | list | None = None,
        neutral_loss_flag: bool = False,
        drop_miss_cleavage_flag: bool = False,
        featured_ions: list[str] | None = None,
        cms2: bool = False,
        task: str = "default",
        output_path: str | Path | None = None,
    ):
        """Initialize a Percolator obj."""
        super().__init__(
            pred_intensities=pred_intensities,
            true_intensities=true_intensities,
            mz=mz,
            xl=("CROSSLINKER_TYPE" in metadata.columns),
            cms2=cms2,
            task=task,
            featured_ions=featured_ions,
            all_features_flag=all_features_flag,
        )

        self.metadata = metadata
        self.output_path = output_path
        self.input_type = input_type
        self.regression_method = regression_method
        self.fdr_cutoff = fdr_cutoff
        self.neutral_loss_flag = neutral_loss_flag
        self.drop_miss_cleavage_flag = drop_miss_cleavage_flag
        self.additional_columns = additional_columns

    @staticmethod
    def unique_peptide_retention_times(psms: pd.DataFrame) -> pd.DataFrame:
        """
        Get one retention time anchor per peptide for the retention time alignment.

        PSMs of the same peptide share the predicted retention time; many identical x values can make the spline fit
        fail and weight frequently observed peptides more. Every peptide is represented by its median observed and
        predicted retention time.

        :param psms: target PSMs with MODIFIED_SEQUENCE, RETENTION_TIME and PREDICTED_IRT
        :return: DataFrame with RETENTION_TIME and PREDICTED_IRT, one row per MODIFIED_SEQUENCE
        """
        return psms.groupby("MODIFIED_SEQUENCE", observed=True)[["RETENTION_TIME", "PREDICTED_IRT"]].median()

    @staticmethod
    def sample_balanced_over_bins(retention_time_df: pd.DataFrame, sample_size: int = 5000) -> pd.Index:
        """
        Sample balanced over bins.

        :param retention_time_df: DataFrame with observed and predicted retention times
        :param sample_size: number of samples
        :return: RT Index
        """
        # Calculate bin edges using Freedman–Diaconis rule
        min_rt = retention_time_df["RETENTION_TIME"].min() * 0.99
        max_rt = retention_time_df["RETENTION_TIME"].max() * 1.01
        bin_width = (
            2
            * scipy.stats.iqr(retention_time_df["RETENTION_TIME"])
            / len(retention_time_df["RETENTION_TIME"]) ** (1 / 3)
        )
        break_points = np.arange(min_rt, max_rt, bin_width)

        retention_time_df["rt_bin_index"] = np.digitize(retention_time_df["RETENTION_TIME"], break_points)

        # sample a subset in each bin. Arbitrary target is 5000 datapoints spread over the bin counts
        points_per_bin = int(np.floor(sample_size / len(break_points)))

        retention_time_df = retention_time_df.groupby("rt_bin_index", group_keys=False).apply(
            lambda x: x.sample(n=min(points_per_bin, len(x)), replace=False)
        )
        return retention_time_df.index

    @staticmethod  # noqa S112, B036
    def get_aligned_predicted_retention_times(
        observed_retention_times_fdr_filtered: np.ndarray | pd.Series,
        predicted_retention_times_fdr_filtered: np.ndarray | pd.Series,
        predicted_retention_times_all: np.ndarray | pd.Series,
        curve_fitting_method: str = "lowess",
        output_path: str | Path | None = None,
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
        total_discarded_percentage = 0.1
        skip_alignment = False
        # the alignment model is cached next to the features file, e.g. <raw>.rescore.tab -> <raw>_alignment_model.pkl
        pickle_model_path = None
        if output_path is not None:
            output_name = Path(output_path).name
            base_name = (
                output_name[: -len(".rescore.tab")] if output_name.endswith(".rescore.tab") else Path(output_name).stem
            )
            pickle_model_path = Path(output_path).parent / f"{base_name}_alignment_model.pkl"
        model_bundle = None
        if pickle_model_path is not None and os.path.isfile(pickle_model_path):
            with open(pickle_model_path, "rb") as f:
                model_bundle = pickle.load(f)
                params = model_bundle["spline_params"]
            skip_alignment = True

        while (
            total_discarded_percentage < 40.0
            and len(observed_rts) >= 5
            and median_abs_error > 0.02
            and not skip_alignment
        ):
            try:
                params = fit_func(predicted_rts, observed_rts)
            except ValueError:
                continue
            aligned_rts_predicted = params[0]

            abs_errors = np.abs(aligned_rts_predicted - observed_rts)
            cut_off = np.percentile(abs_errors, 100 - discard_percentage)
            median_abs_error = np.median(np.abs(abs_errors))

            logger.debug(f"Median absolute error aligned rts: {median_abs_error}")

            if median_abs_error > 0.02:
                keep_idxs = np.nonzero(abs_errors < cut_off)
                observed_rts = observed_rts[keep_idxs[0]]
                predicted_rts = predicted_rts[keep_idxs[0]]

                discard_percentage *= 1.2
                total_discarded_percentage += discard_percentage

        if (len(observed_rts) < 3 or "params" not in locals()) and model_bundle is None:
            return predicted_retention_times_all

        logger.debug(f"Observed RT anchor points:\n{observed_retention_times_fdr_filtered}")
        logger.debug(f"Predicted RT anchor points:\n{predicted_retention_times_fdr_filtered}")

        if curve_fitting_method == "spline":
            if model_bundle is not None:
                spline_params = model_bundle["spline_params"]
            else:
                spline_params = params[1:]  # (t, c, k)
            model_bundle = {
                "curve_fitting_method": "spline",
                "spline_params": spline_params,
            }
            aligned_rts_predicted = interpolate.BSpline(*spline_params)(predicted_retention_times_all)
        elif curve_fitting_method == "lowess":
            lowess_model = lowess.Lowess()
            lowess_model.fit(predicted_rts, observed_rts, frac=frac, robust_iters=it)
            aligned_rts_predicted = lowess_model.predict(np.array(predicted_retention_times_all))
        else:  # logistic
            aligned_rts_predicted = logistic(
                predicted_retention_times_all, *opt.curve_fit(logistic, predicted_rts, observed_rts, method="lm")[0]
            )

        if pickle_model_path is not None and model_bundle is not None:
            with open(pickle_model_path, "wb") as f:
                pickle.dump(model_bundle, f)

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
        if scores_df["ScanNr"].nunique() == 1:
            raise NotImplementedError

        return (
            scores_df.sort_values(by=[scoring_feature])
            .groupby("ScanNr")[scoring_feature]
            .diff(periods=1)
            .fillna(0)
            .sort_index()
            .to_numpy()
        )

    @staticmethod
    def get_specid(metadata_subset: pd.Series | tuple) -> str:
        """
        Create a unique identifier used as spectrum id in percolator, this is not parsed by percolator but functions \
        as a key to map percolator results back to our internal representation.

        :param metadata_subset: tuple of (raw_file, scan_number, modified_sequence,
            charge and optionally scan_event_number)
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
    def get_ptms(peptides: pd.Series, standard_unimod_ids: tuple[int, ...] = STANDARD_UNIMOD_IDS) -> pd.Series:
        """
        Get the PTMs of every peptide, without the standard modifications.

        :param peptides: modified peptide sequences with UNIMOD annotations, e.g. "_.M[UNIMOD:35]PEPS[UNIMOD:21]K._"
        :param standard_unimod_ids: UNIMOD ids of fixed modifications and labels that are not counted as PTMs
        :return: the set of UNIMOD ids (as strings) of the PTMs of every peptide
        """
        standard = {str(unimod_id) for unimod_id in standard_unimod_ids}
        return peptides.map(lambda peptide: set(re.findall(r"UNIMOD:(\d+)\]", peptide)) - standard)

    @staticmethod
    def get_ptm_indicators(
        peptides: pd.Series, standard_unimod_ids: tuple[int, ...] = STANDARD_UNIMOD_IDS
    ) -> pd.DataFrame:
        """
        Get a 0/1 feature for every PTM carried by any of the peptides.

        Like the variable modification columns of MSFragger, the features let the FDR estimation learn how reliable the
        identifications of each PTM are.

        :param peptides: modified peptide sequences with UNIMOD annotations
        :param standard_unimod_ids: UNIMOD ids of fixed modifications and labels that are not counted as PTMs
        :return: one column ptm_UNIMOD_<id> per PTM, 1 if the peptide carries the PTM, else 0
        """
        ptms = Percolator.get_ptms(peptides, standard_unimod_ids)
        unimod_ids = sorted(set().union(*ptms), key=int)
        return pd.DataFrame(
            {f"ptm_UNIMOD_{unimod_id}": ptms.map(lambda p, u=unimod_id: int(u in p)) for unimod_id in unimod_ids},
            index=peptides.index,
        )

    @staticmethod
    def calculate_mass_difference(metadata_subset: tuple[float, float]) -> float:
        """
        Calculate mass difference.

        :param metadata_subset: experimental and calculated mass as tuple
        :return: mass difference
        """
        experimental_mass, calculated_mass = metadata_subset
        return calculated_mass - experimental_mass

    @staticmethod
    def calculate_mass_difference_ppm(metadata_subset: tuple[float, float]) -> float:
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
        if self.xl:
            self.metrics_val["missedCleavages_A"] = self.metadata["SEQUENCE_A"].apply(Percolator.count_missed_cleavages)
            self.metrics_val["missedCleavages_B"] = self.metadata["SEQUENCE_B"].apply(Percolator.count_missed_cleavages)
            self.metrics_val["KR_A"] = self.metadata["SEQUENCE_A"].apply(Percolator.count_arginines_and_lysines)
            self.metrics_val["KR_B"] = self.metadata["SEQUENCE_B"].apply(Percolator.count_arginines_and_lysines)
            self.metrics_val["sequence_length_a"] = self.metadata["SEQUENCE_A"].apply(lambda x: len(x))
            self.metrics_val["sequence_length_b"] = self.metadata["SEQUENCE_B"].apply(lambda x: len(x))
            for column_name in self.metadata.columns:
                self.metrics_val[column_name] = self.metadata[column_name]
        else:
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
        self.metrics_val["ExpMass"] = (
            self.metadata["MASS"] + self.metadata["PRECURSOR_CHARGE"] * constants.MASSES["PROTON"]
        ) / self.metadata["PRECURSOR_CHARGE"]

    def add_additional_features(self):
        """Add additional features from custom serch results if specified."""
        feature_cols = []

        if isinstance(self.additional_columns, list):
            feature_cols = self.additional_columns
        elif isinstance(self.additional_columns, str) and (self.additional_columns.lower() == "all"):
            feature_cols = [x for x in self.metadata.columns if x.lower() not in set(self.BASE_COLUMNS)]
            feature_cols = [x for x in feature_cols if not x.lower().startswith("unnamed")]  # remove Unnamed cols

        for col in feature_cols:
            if col not in self.metadata.columns:
                raise ValueError(f"provided column: {col} cannot be found in search results.")
            elif not pd.api.types.is_numeric_dtype(self.metadata[col]):
                raise ValueError(f"wrong datatype for column: {col}, datatype must be numerical.")
            else:
                self.metrics_val[col] = self.metadata[col]

    def add_percolator_metadata_columns(self):
        """Add metadata columns needed by percolator, e.g. to identify a PSM."""
        if self.xl:
            spec_id_cols = ["RAW_FILE", "SCAN_NUMBER", "MODIFIED_SEQUENCE_A", "MODIFIED_SEQUENCE_B", "PRECURSOR_CHARGE"]
            modified_sequence_a = self.metadata["MODIFIED_SEQUENCE_A"].astype(str)
            modified_sequence_b = self.metadata["MODIFIED_SEQUENCE_B"].astype(str)
            self.metrics_val["Peptide"] = (modified_sequence_a + "_" + modified_sequence_b).apply(
                lambda x: "_." + x + "._"
            )
            self.metrics_val["Proteins"] = modified_sequence_a + "_" + modified_sequence_b
            self.metrics_val["Label"] = self.target_decoy_labels
        else:
            spec_id_cols = ["RAW_FILE", "SCAN_NUMBER", "MODIFIED_SEQUENCE", "PRECURSOR_CHARGE"]
            self.metrics_val["Peptide"] = self.metadata["MODIFIED_SEQUENCE"].apply(lambda x: "_." + x + "._")
            self.metrics_val["Proteins"] = self.metadata["PROTEINS"]

        if "SCAN_EVENT_NUMBER" in self.metadata.columns:
            spec_id_cols.append("SCAN_EVENT_NUMBER")
        self.metrics_val["SpecId"] = self.metadata[spec_id_cols].apply(Percolator.get_specid, axis=1)
        self.metrics_val["Label"] = self.target_decoy_labels
        self.metrics_val["ScanNr"] = self.metadata["SCAN_NUMBER"]
        self.metrics_val["filename"] = self.metadata["RAW_FILE"]

    def apply_lda_and_get_indices_below_fdr(
        self, initial_scoring_feature: str = "spectral_angle", fdr_cutoff: float = 0.01
    ):  # noqa: B036
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

        scores_df["fdr"] = Percolator.calculate_fdrs(scores_df["Label"])
        # filter for targets only
        scores_df = scores_df[scores_df["Label"] == TargetDecoyLabel.TARGET]

        accepted_indices = scores_df.index[scores_df["fdr"] < fdr_cutoff]
        if len(accepted_indices) == 0:
            logger.warning(
                f"Could not find any targets below {fdr_cutoff} out of {len(scores_df.index)} "
                "targets in total. Retrying with higher FDR cutoff..."
            )
            return np.array([])

        logger.info(
            f"Found {len(accepted_indices)} (out of {len(scores_df.index)}) targets below {fdr_cutoff} \
            FDR using {feature_name} as feature"
        )

        return np.sort(scores_df.index[: len(accepted_indices)])

    @staticmethod
    def calculate_fdrs(sorted_labels: pd.Series | np.ndarray) -> np.ndarray:
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
        first_columns = ["SpecId", "Label", "ScanNr", "filename", "ExpMass"]
        last_columns = ["Peptide", "Proteins"]
        mid_columns = list(set(all_columns) - set(first_columns) - set(last_columns))
        new_columns = first_columns + sorted(mid_columns) + last_columns
        self.metrics_val = self.metrics_val[new_columns]

    @staticmethod
    def create_modified_ions_mask(starting_index):
        """
        Create modified ions mask starting from the given index.

        :param starting_index: starting index
        :return: modified ions mask
        """
        total_length = 174
        array = np.zeros(total_length)
        array[starting_index:] = 1
        return array

    def _find_phospho_positions(self):
        all_positions = []
        new_phoshpo_sequences = internal_without_mods_keep_loc(self.metadata["MODIFIED_SEQUENCE"].values, "21")
        for peptide in new_phoshpo_sequences:
            positions = []
            index = 0
            while "[UNIMOD:21]" in peptide:
                index = peptide.find("[UNIMOD:21]", index)
                if index == -1:
                    break
                positions.append(index - peptide[:index].count("[UNIMOD:21]") * len("[UNIMOD:21]"))
                index += len("[UNIMOD:21]")
            all_positions.append(positions)
        return all_positions

    def get_first_b_and_y_modified_ion(self):
        """Get first b and y modified ion masks for phospho peptides.

        :return: b and y ions masks
        """
        all_phospho_sites_positions = self._find_phospho_positions()
        b_ions_mask = []
        y_ions_mask = []
        for position, length in zip(all_phospho_sites_positions, self.metrics_val["sequence_length"], strict=True):
            if len(position) == 0:
                b_ions_mask.append(self.ion_mask["b"])
                y_ions_mask.append(self.ion_mask["y"])
            else:
                mod_b_ions_mask = Percolator.create_modified_ions_mask(position[0] * 6 - 3)
                b_ions_mask.append(self.ion_mask["b"] * mod_b_ions_mask)
                mod_y_ions_mask = Percolator.create_modified_ions_mask((length - position[-1]) * 6)
                y_ions_mask.append(self.ion_mask["y"] * mod_y_ions_mask)

        return b_ions_mask, y_ions_mask

    def get_site_determining_ion_masks(self, possible_sites: str = "STY"):
        """Get b and y ion masks of the site determining ions of phospho peptides.

        Site determining ions distinguish a localization from the closest alternative: for every phosphorylated
        residue, the b and y ions that contain either the phosphorylated residue or the closest unmodified residue
        that could carry the phosphorylation, but not both. Peptides without phosphorylation or without an
        alternative site get empty masks.

        :param possible_sites: residues that can carry the phosphorylation
        :return: b and y ions masks
        """
        # every fragment number (1-based) has 6 ions: y (charge 1-3), b (charge 1-3)
        n_ions = len(self.ion_mask["b"])
        max_fragment = n_ions // 6
        b_ions_mask, y_ions_mask = [], []
        sequences = internal_without_mods_keep_loc(self.metadata["MODIFIED_SEQUENCE"].values, "21")
        for sequence in sequences:
            residues = re.findall(r"([A-Z])((?:\[UNIMOD:21\])?)", sequence)
            length = len(residues)
            sites = [i + 1 for i, (_, mod) in enumerate(residues) if mod]
            alternatives = [i + 1 for i, (aa, mod) in enumerate(residues) if aa in possible_sites and not mod]
            b_mask, y_mask = np.zeros(n_ions), np.zeros(n_ions)
            for site in sites:
                if not alternatives:
                    break
                closest = min(alternatives, key=lambda alternative: (abs(alternative - site), alternative))  # noqa: B023
                for fragment in range(min(site, closest), max(site, closest)):
                    # b ion b_k and y ion y_(length - k) contain exactly one of the two residues
                    if fragment <= max_fragment:
                        b_mask[(fragment - 1) * 6 : fragment * 6] = 1
                    if 1 <= length - fragment <= max_fragment:
                        y_mask[(length - fragment - 1) * 6 : (length - fragment) * 6] = 1
            b_ions_mask.append(self.ion_mask["b"] * b_mask)
            y_ions_mask.append(self.ion_mask["y"] * y_mask)
        return b_ions_mask, y_ions_mask

    def _deduplicate_intensities(self, mz, intensities):
        """Take highest intensity prediction for ions with same mz, return new array."""
        n, m = intensities.shape
        mz_row = mz.toarray()
        intensity_row = intensities.toarray()

        for row in range(n):
            u, c = np.unique(mz_row[row], return_counts=True)
            dup = u[(c > 1) & (u > constants.EPSILON)]
            for val in dup:
                mask = mz_row[row] == val
                if np.any(mask):
                    temp_max_idx = np.argmax(intensity_row[row][mask])
                    indices = np.where(mask)[0]
                    max_idx = indices[temp_max_idx]
                    other_indices = indices[indices != max_idx]
                    intensities[row, other_indices] = 0.0

        return intensities

    def calc(self):  # noqa: C901
        """Adds percolator metadata and feature columns to metrics_val based on PSM metadata."""
        if self.task == "multifrag":
            self.pred_intensities = self._deduplicate_intensities(self.mz, self.pred_intensities)

        self.add_common_features()
        self.target_decoy_labels = self.metadata["REVERSE"].apply(Percolator.get_target_decoy_label).to_numpy()
        np.random.seed(1)
        # add Prosit or Andromeda features
        self.add_additional_features()
        """aa_mod_combinations = []
        for seq in self.metadata['MODIFIED_SEQUENCE']:
            aa_mod_combinations.append(extract_aa_mod(seq))"""
        # aa_mod_combinations = get_mod_comb(constants.MOD_COMB)
        # for aa_mod in aa_mod_combinations:
        #     self.metrics_val[aa_mod] = self.metadata["MODIFIED_SEQUENCE"].apply(
        #         lambda x: x.count(aa_mod) if aa_mod in x else 0  # noqa: B023
        #     )
        # self.metadata['AA_MOD_COMBINATIONS'] = aa_mod_combinations
        # all_combinations = set([item for sublist in self.metadata['AA_MOD_COMBINATIONS'] for item in sublist])

        if self.input_type == "rescore":
            # add additional features
            # most intense annotated peak per spectrum, 1 if not annotated (e.g. annotated with older versions)
            if "MOST_INTESE_PEAK" in self.metadata.columns:
                self.metrics_val["MOST_INTESE_PEAK"] = self.metadata["MOST_INTESE_PEAK"].fillna(1)
            else:
                self.metrics_val["MOST_INTESE_PEAK"] = 1.0
            fragments_ratio = fr.FragmentsRatio(
                self.pred_intensities,
                self.true_intensities,
                xl=self.xl,
                cms2=self.cms2,
                task=self.task,
                featured_ions=self.featured_ions,
                most_intense_peaks=self.metrics_val["MOST_INTESE_PEAK"].values,
            )
            fragments_ratio.calc()
            if self.neutral_loss_flag:
                b_ions_mask, y_ions_mask = self.get_first_b_and_y_modified_ion()
                fragments_ratio.calc(b_mask_mod=np.array(b_ions_mask), y_mask_mod=np.array(y_ions_mask), suffix="_mod")
                if self.task == "default" and not self.cms2:
                    # site determining ions: the ions that distinguish a localization from the closest alternative
                    b_ions_mask, y_ions_mask = self.get_site_determining_ion_masks()
                    fragments_ratio.calc(
                        b_mask_mod=np.array(b_ions_mask), y_mask_mod=np.array(y_ions_mask), suffix="_sdi"
                    )
            similarity = sim.SimilarityMetrics(
                self.pred_intensities,
                self.true_intensities,
                self.mz,
                xl=self.xl,
                cms2=self.cms2,
                all_features_flag=self.all_features_flag,
            )
            similarity.calc()

            self.metrics_val = pd.concat(
                [self.metrics_val, fragments_ratio.metrics_val, similarity.metrics_val], axis=1
            )
            if self.neutral_loss_flag:
                self.metrics_val["ANNOTATED_NL_COUNT"] = self.metadata["ANNOTATED_NL_COUNT"]
                self.metrics_val["EXPECTED_NL_COUNT"] = self.metadata["EXPECTED_NL_COUNT"]
                self.metrics_val["NL_FRACTION"] = self.metrics_val.apply(
                    lambda x: x["ANNOTATED_NL_COUNT"] / x["EXPECTED_NL_COUNT"] if x["EXPECTED_NL_COUNT"] > 0 else 0,
                    axis=1,
                )
                self.metrics_val["NL_NOT_POSSIBLE"] = self.metrics_val.apply(
                    lambda x: 1 if x["EXPECTED_NL_COUNT"] == 0 else 0, axis=1
                )
                if "EXPECTED_NL_COUNT_PLAIN" in self.metadata.columns:
                    # neutral loss features of plain decoys: without the neutral loss on the decoy site
                    annotated = self.metadata["ANNOTATED_NL_COUNT_PLAIN"].to_numpy(dtype=float)
                    expected = self.metadata["EXPECTED_NL_COUNT_PLAIN"].to_numpy(dtype=float)
                    self.metrics_val["NL_FRACTION_PLAIN"] = np.where(
                        expected > 0, annotated / np.maximum(expected, 1), 0
                    )
                    self.metrics_val["NL_NOT_POSSIBLE_PLAIN"] = (expected == 0).astype(int)
                # columns_to_remove = []
                # for col in self.metrics_val.columns:
                #    if "vs_predicted" in col:
                #        columns_to_remove.append(col)
                # self.metrics_val.drop(columns=columns_to_remove, inplace=True)
            if self.drop_miss_cleavage_flag:
                self.metrics_val.drop(columns=["missedCleavages", "KR"], inplace=True)
            if self.xl:
                self.metrics_val["collision_energy_aligned"] = self.metadata["COLLISION_ENERGY"] / 100.0
            else:
                lda_failed = False
                try:
                    idxs_below_lda_fdr = self.apply_lda_and_get_indices_below_fdr(fdr_cutoff=self.fdr_cutoff)
                except ValueError:
                    lda_failed = True
                    pass
                current_fdr = self.fdr_cutoff
                while (
                    not lda_failed
                    and len(idxs_below_lda_fdr) <= 200
                    and (len(idxs_below_lda_fdr) / len(self.target_decoy_labels)) < 0.5
                ):
                    current_fdr += 0.01
                    idxs_below_lda_fdr = self.apply_lda_and_get_indices_below_fdr(fdr_cutoff=current_fdr)
                    if current_fdr >= 0.1:
                        lda_failed = True
                        break

                if len(self.metadata[self.metadata["PROTEINS"].str.contains("QC:")]) > 0:
                    lda_filter = (
                        self.metadata["MODIFIED_SEQUENCE"].str.match(r"^(?!.*UNIMOD(?!:737|:411|:4|:267|:35)).*$")
                    ) & (self.metadata["PROTEINS"].str.contains("QC:") & (~self.metadata["REVERSE"]))
                else:
                    lda_filter = (
                        self.metadata["MODIFIED_SEQUENCE"].str.match(r"^(?!.*UNIMOD(?!:737|:411|:4|:267|:35)).*$")
                    ) & (~self.metadata["REVERSE"])

                if lda_failed:
                    anchors = Percolator.unique_peptide_retention_times(self.metadata[lda_filter])
                    sampled_idxs = Percolator.sample_balanced_over_bins(anchors)
                else:
                    try:
                        anchors = Percolator.unique_peptide_retention_times(
                            self.metadata.iloc[idxs_below_lda_fdr, :][lda_filter]
                        )
                        sampled_idxs = Percolator.sample_balanced_over_bins(anchors)
                    except ValueError:
                        lda_failed = True
                        anchors = Percolator.unique_peptide_retention_times(self.metadata[lda_filter])
                        sampled_idxs = anchors.index

                # the fit uses one point per peptide; the aligned retention times are then predicted for all PSMs
                file_sample = anchors.loc[sampled_idxs].sort_values("PREDICTED_IRT")

                aligned_predicted_rts = Percolator.get_aligned_predicted_retention_times(
                    file_sample["RETENTION_TIME"],
                    file_sample["PREDICTED_IRT"],
                    self.metadata["PREDICTED_IRT"],
                    self.regression_method,
                    self.output_path,
                )

                # self.metrics_val["RT"] = self.metadata["RETENTION_TIME"]
                # self.metrics_val["pred_RT"] = self.metadata["PREDICTED_IRT"]
                # self.metrics_val["iRT"] = aligned_predicted_rts
                self.metrics_val["abs_rt_diff"] = np.abs(self.metadata["RETENTION_TIME"] - aligned_predicted_rts)

                # unmod_rescore_dir = new_rescore_dir.replace('localize_mod','score_unmod')

                # the intensity features are relative to the most intense annotated peak of each candidate, which
                # differs between the localization candidates of a spectrum. Convert them back to the measured
                # intensities (log10), so they are comparable between candidates and spectra.
                # The relative intensities of the modified ions are kept as additional features, since they separate
                # candidates with adjacent sites better.
                for feature in ["observed_intensity_mod", "sum_observed_and_predicted_mod"]:
                    if feature in self.metrics_val.columns:
                        self.metrics_val[f"{feature}_relative"] = self.metrics_val[feature]
                for feature in [
                    "observed_intensity",
                    "sum_observed_and_predicted",
                    "observed_intensity_mod",
                    "sum_observed_and_predicted_mod",
                ]:
                    if feature not in self.metrics_val.columns:
                        continue
                    measured = self.metrics_val[feature] * self.metrics_val["MOST_INTESE_PEAK"]
                    self.metrics_val[feature] = np.log10(measured.where(measured > 0, 1.0))

                if lda_failed:
                    self.metrics_val["lda_scores"] = 0
                    median_abs_error = np.median(self.metrics_val["abs_rt_diff"])
                    delta_95_error = np.percentile(self.metrics_val["abs_rt_diff"], 95)
                else:
                    median_abs_error = np.median(self.metrics_val["abs_rt_diff"].iloc[idxs_below_lda_fdr][lda_filter])
                    delta_95_error = np.percentile(
                        self.metrics_val["abs_rt_diff"].iloc[idxs_below_lda_fdr][lda_filter], 95
                    )
                self.metrics_val.drop("lda_scores", axis=1, inplace=True)

                logger.info(
                    "Median absolute error predicted vs observed retention time on targets < 1% FDR: "
                    f"{median_abs_error}"
                )
                logger.info(
                    "Delta 95 absolute error predicted vs observed retention time on targets < 1% FDR: "
                    f"{delta_95_error}"
                )
                self.metrics_val["andromeda"] = self.metadata["SCORE"]
            self.metrics_val.drop(columns=["MOST_INTESE_PEAK"], inplace=True)
        else:
            self.metrics_val["andromeda"] = self.metadata["SCORE"]

        """for aa_comb in all_combinations:
            aa_comb_values = self.metadata['AA_MOD_COMBINATIONS'].apply(
                lambda x: x.count(aa_comb) if aa_comb in x else 0
            )
            self.metrics_val[aa_comb] = aa_comb_values"""

        if "Annotated_Ions_MSF" in self.metadata.columns:
            self.metrics_val["annotated_ions"] = self.metadata["Annotated_Ions_MSF"]
            self.metrics_val["delta_mass_ppm"] = abs(self.metadata["MZ_diff_MSF"] * 1000000 / self.metrics_val["Mass"])
            self.metrics_val["next_score"] = self.metadata["NEXT_SCORE"]
            self.metrics_val["log10_evalue"] = self.metadata["EXPECT"].apply(lambda x: math.log10(x))

        self.add_percolator_metadata_columns()

        if self.input_type == "rescore":
            # TODO: only add this feature if they are not all zero
            # self.metrics_val['spectral_angle_delta_score'] = Percolator.get_delta_score(self.metrics_val[['ScanNr',
            # 'spectral_angle']], 'spectral_angle')
            # self.metrics_val.drop(columns=['sum_observed_and_predicted','observed_intensity'],inplace=True)
            pass
        else:
            self.metrics_val["andromeda_delta_score"] = Percolator.get_delta_score(
                self.metrics_val[["ScanNr", "andromeda"]], "andromeda"
            )
        # the calculated mass is only a feature of linear peptides (add_common_features)
        self.metrics_val.drop(columns="Mass", errors="ignore", inplace=True)

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
    if q_knots[0] == 0:
        q_knots[0] += 1e-6
    if q_knots[0] == q_knots[1]:
        q_knots[1] += 1e-6
    t, c, k = interpolate.splrep(x, y, t=q_knots, s=2)
    yfit = interpolate.BSpline(t, c, k)(x)
    return yfit, t, c, k


def logistic(x: pd.Series | np.ndarray, a: float, b: float, c: float, d: float):
    """Calculates logistic regression function."""
    exponent = np.clip(-c * (x - d), -700, 700)  # make this stable, i.e. avoid 0.0 or inf
    return a / (1.0 + np.exp(exponent)) + b
