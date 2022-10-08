import hashlib
import logging
import enum

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.stats

from .metric import Metric
from . import fragments_ratio as fr
from . import similarity as sim

logger = logging.getLogger(__name__)


class TargetDecoyLabel(enum.IntEnum):
    """
    Target and decoy labels as used by Percolator
    """
    TARGET = 1
    DECOY = -1


class Percolator(Metric):
    """
    Expects the following metadata columns:
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
    target_decoy_labels: np.array
    input_type: str
    fdr_cutoff: float

    def __init__(self, metadata: pd.DataFrame, pred_intensities, true_intensities, input_type, all_features_flag=False, fdr_cutoff=0.01):
        self.metadata = metadata
        self.input_type = input_type
        self.all_features_flag = all_features_flag
        self.fdr_cutoff = fdr_cutoff
        super().__init__(pred_intensities, true_intensities)

    @staticmethod
    def sample_balanced_over_bins(retention_time_df, sample_size: int = 5000):

        # bin retention times
        #print(retention_time_df['RETENTION_TIME'])
        min_rt = retention_time_df['RETENTION_TIME'].min() * 0.99
        max_rt = retention_time_df['RETENTION_TIME'].max() * 1.01
        bin_width = 2 * scipy.stats.iqr(retention_time_df['RETENTION_TIME']) / len(retention_time_df['RETENTION_TIME']) ** (
                1 / 3)  # Freedman–Diaconis rule
        break_points = np.arange(min_rt, max_rt, bin_width)
        retention_time_df['rt_bin_index'] = np.digitize(retention_time_df['RETENTION_TIME'], break_points)

        # sample a subset in each bin. Arbitrary target is 5000 datapoints spread over the bin counts
        points_per_bin = int(np.floor(sample_size / len(break_points)))
        retention_time_df = retention_time_df.groupby('rt_bin_index').apply(
            lambda x: pd.DataFrame.sample(x, n=min(points_per_bin, len(x)), replace=False))
        return retention_time_df.reset_index(level=0, drop=True).index

    @staticmethod
    def get_aligned_predicted_retention_times(observed_retention_times_fdr_filtered, predicted_retention_times_fdr_filtered,
                                              predicted_retention_times_all):
        """
        Apply loess regression to find a mapping from predicted iRT values to experimental retention times
        """
        
        observed_rts = np.array(observed_retention_times_fdr_filtered, dtype=np.float64)
        predicted_rts = np.array(predicted_retention_times_fdr_filtered, dtype=np.float64)
        
        # TODO: use Akaike information criterion to choose a good value for frac
        frac = 0.5 # Between 0 and 1. The fraction of the data used when estimating each y-value.
        it = 0 # The number of residual-based reweightings to perform. Don't use the iterative reweighting (it > 1), this result in NaNs
        discard_percentage = 0.1 # in percents, so 0.1 = 0.1% (not 10%!)
        while discard_percentage < 50.0:
            aligned_rts_predicted = lowess(observed_rts, 
                                           predicted_rts,
                                           frac=frac, it=it, return_sorted=False)
            abs_errors = np.abs(aligned_rts_predicted - observed_rts)
            cut_off = np.percentile(abs_errors, 100 - discard_percentage)
            median_abs_error = np.median(np.abs(abs_errors))
            logger.debug(f"Median absolute error aligned rts: {median_abs_error}")
            
            if median_abs_error > 0.02:
                keep_idxs = np.nonzero(abs_errors < cut_off)
                observed_rts = observed_rts[keep_idxs[0]]
                predicted_rts = predicted_rts[keep_idxs[0]]
            else:
                break
            
            discard_percentage *= 1.5
        
        logger.debug(f"Observed RT anchor points:\n{observed_retention_times_fdr_filtered}")
        logger.debug(f"Predicted RT anchor points:\n{predicted_retention_times_fdr_filtered}")
        
        # TODO; test for NaNs and use interpolation to fill them up
        aligned_rts_predicted = lowess(observed_rts, 
                                       predicted_rts,
                                       xvals=predicted_retention_times_all.astype(np.float64), 
                                       frac=frac, it=it)
        return aligned_rts_predicted

    @staticmethod
    def get_scannr(metadata_subset):
        """
        Creates a hash of the raw_file and scan number to use as a unique scan number in percolator
        :param metadata_subset: tuple of (raw_file, scan_number)
        :return: hashed unique id
        """
        raw_file, scan_number = metadata_subset
        s = "{}{}".format(raw_file, scan_number).encode()
        return int(hashlib.sha224(s).hexdigest()[:12], 16)

    @staticmethod
    def get_delta_score(
            scores_df: pd.DataFrame,
            scoring_feature: str
    ):
        """
        Calculates delta scores by sorting (from high to low) and grouping PSMs by scan number. Inside each group the delta scores are
        calculated per PSM to the next best of that group. The lowest scoring PSM of each group receives a delta score of 0.
        :param scores_df: must contain two columns: scoring_feature (eg. 'spectral_angle') and 'ScanNr'
        :param scoring_feature: feature name to get the delta scores of
        :return: numpy array of delta scores
        """
        # TODO: sort after grouping for better efficiency
        scores_df = scores_df.sort_values(by=scoring_feature, ascending=True)
        groups = scores_df.groupby(["ScanNr"])
        t = groups.apply(lambda scores_df_: scores_df_[scoring_feature] - scores_df_[scoring_feature].shift(1))
        # apply doesnt work for one group only
        if len(groups) == 1:
            raise NotImplementedError
        scores_df['delta_' + scoring_feature] = pd.Series(t.reset_index(level=0, drop=True))
        scores_df.fillna(0, inplace=True)
        scores_df.sort_index(inplace=True)
        return scores_df['delta_' + scoring_feature].to_numpy()

    @staticmethod
    def get_specid(metadata_subset):
        """
        Create a unique identifier used as spectrum id in percolator, this is not parsed by percolator but functions as a key to map percolator results back to our internal representation
        :param metadata_subset: tuple of (raw_file, scan_number, modified_sequence, charge)
        :return: percolator spectrum id
        """
        raw_file, scan_number, modified_sequence, charge, scan_event_number = metadata_subset
        s = "{}-{}-{}-{}-{}".format(
            raw_file, scan_number, modified_sequence, charge, scan_event_number
        )
        return s

    @staticmethod
    def count_missed_cleavages(sequence):
        """
        Count number of missed cleavages assuming Trypsin/P proteolysis
        :param sequence:
        """
        return sequence[:-1].count("K") + sequence[:-1].count("R")

    @staticmethod
    def count_arginines_and_lysines(sequence):
        return sequence.count("K") + sequence.count("R")

    @staticmethod
    def calculate_mass_difference(metadata_subset):
        experimental_mass, calculated_mass = metadata_subset
        return calculated_mass - experimental_mass

    @staticmethod
    def calculate_mass_difference_ppm(metadata_subset):
        experimental_mass, calculated_mass = metadata_subset
        return (calculated_mass - experimental_mass) / experimental_mass * 1e6

    @staticmethod
    def get_target_decoy_label(reverse):
        """
        :return: target/decoy label for percolator, 1 = Target, -1 = Decoy
        """
        return TargetDecoyLabel.DECOY if reverse else TargetDecoyLabel.TARGET

    def add_common_features(self):
        """
        Add features used by both Andromeda and Prosit feature scoring sets
        """
        self.metrics_val['missedCleavages'] = self.metadata['SEQUENCE'].apply(Percolator.count_missed_cleavages)
        self.metrics_val['KR'] = self.metadata['SEQUENCE'].apply(Percolator.count_arginines_and_lysines)
        self.metrics_val['sequence_length'] = self.metadata['SEQUENCE'].apply(lambda x: len(x))

        self.metrics_val['Mass'] = self.metadata['CALCULATED_MASS']  # this is the calculated mass used as a feature
        
        # for now, disable delta mass features as MaxQuant does not seem to provide the 
        # experimental mass in msms.txt. Both the Mass and m/z columns are theoretical masses
        #self.metrics_val['deltaM_Da'] = self.metadata[['MASS', 'CALCULATED_MASS']].apply(Percolator.calculate_mass_difference, axis=1)
        #self.metrics_val['absDeltaM_Da'] = np.abs(self.metrics_val['deltaM_Da'])
        #self.metrics_val['deltaM_ppm'] = self.metadata[['MASS', 'CALCULATED_MASS']].apply(Percolator.calculate_mass_difference_ppm, axis=1)
        #self.metrics_val['absDeltaM_ppm'] = np.abs(self.metrics_val['deltaM_ppm'])

        self.metrics_val['Charge1'] = (self.metadata['PRECURSOR_CHARGE'] == 1).astype(int)
        self.metrics_val['Charge2'] = (self.metadata['PRECURSOR_CHARGE'] == 2).astype(int)
        self.metrics_val['Charge3'] = (self.metadata['PRECURSOR_CHARGE'] == 3).astype(int)
        self.metrics_val['Charge4'] = (self.metadata['PRECURSOR_CHARGE'] == 4).astype(int)
        self.metrics_val['Charge5'] = (self.metadata['PRECURSOR_CHARGE'] == 5).astype(int)
        self.metrics_val['Charge6'] = (self.metadata['PRECURSOR_CHARGE'] == 6).astype(int)

        self.metrics_val['UnknownFragmentationMethod'] = (~self.metadata['FRAGMENTATION'].isin(['HCD', 'CID'])).astype(int)
        self.metrics_val['HCD'] = (self.metadata['FRAGMENTATION'] == 'HCD').astype(int)
        self.metrics_val['CID'] = (self.metadata['FRAGMENTATION'] == 'CID').astype(int)

    def add_percolator_metadata_columns(self):
        """
        Add metadata columns needed by percolator, e.g. to identify a PSM
        """
        self.metrics_val['SpecId'] = self.metadata[['RAW_FILE', 
                                                    'SCAN_NUMBER', 
                                                    'MODIFIED_SEQUENCE', 
                                                    'PRECURSOR_CHARGE', 
                                                    'SCAN_EVENT_NUMBER']].apply(Percolator.get_specid, axis=1)
        self.metrics_val['Label'] = self.target_decoy_labels
        self.metrics_val['ScanNr'] = self.metadata[['RAW_FILE', 'SCAN_NUMBER']].apply(Percolator.get_scannr, axis=1)

        #self.metrics_val['ExpMass'] = self.metadata['MASS']
        self.metrics_val['Peptide'] = self.metadata['MODIFIED_SEQUENCE'].apply(lambda x: '_.' + x + '._')
        self.metrics_val['Protein'] = self.metadata['MODIFIED_SEQUENCE']  # we don't need the protein ID to get PSM / peptide results, fill with peptide sequence

    def apply_lda_and_get_indices_below_fdr(self, initial_scoring_feature='spectral_angle', fdr_cutoff=0.01):
        """
        Applies a linear discriminant analysis on the features calculated so far (before retention time alignment) to estimate false discovery rates (FDRs).
        """
        target_idxs_below_fdr = self.get_indices_below_fdr(initial_scoring_feature, fdr_cutoff=fdr_cutoff)
        decoy_idxs = np.argwhere(self.target_decoy_labels == TargetDecoyLabel.DECOY).flatten()
        logger.info(f"Found {len(target_idxs_below_fdr)} targets and {len(decoy_idxs)} decoys as input for the LDA model")

        lda_idxs = np.concatenate((target_idxs_below_fdr, decoy_idxs)).astype(int)

        X = self.metrics_val.iloc[lda_idxs, :].to_numpy()
        y = self.target_decoy_labels[lda_idxs]

        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)

        self.metrics_val['lda_scores'] = lda.decision_function(self.metrics_val.to_numpy())

        return self.get_indices_below_fdr('lda_scores', fdr_cutoff=fdr_cutoff)

    def get_indices_below_fdr(self, feature_name, fdr_cutoff=0.01):
        scores_df = self.metrics_val[[feature_name]].copy()
        scores_df['Label'] = self.target_decoy_labels
        #scores_df['Sequence'] = self.metadata['SEQUENCE']
        scores_df = scores_df.sort_values(feature_name, ascending=False)
        logger.debug(scores_df.head(100))
        
        scores_df['fdr'] = Percolator.calculate_fdrs(scores_df['Label'])

        # filter for targets only
        scores_df = scores_df[scores_df['Label'] == TargetDecoyLabel.TARGET]

        accepted_indices = scores_df.index[scores_df['fdr'] < fdr_cutoff]
        if len(accepted_indices) == 0:
            logger.error(f"Could not find any targets below {fdr_cutoff} out of {len(scores_df.index)} targets in total")
            return np.array([])
        
        logger.info(f"Found {len(accepted_indices)} (out of {len(scores_df.index)}) targets below {fdr_cutoff} FDR using {feature_name} as feature")
        
        return np.sort(scores_df.index[:len(accepted_indices)])

    @staticmethod
    def calculate_fdrs(sorted_labels):
        cumulative_decoy_count = np.cumsum(sorted_labels == TargetDecoyLabel.DECOY) + 1
        cumulative_target_count = np.cumsum(sorted_labels == TargetDecoyLabel.TARGET) + 1
        return Percolator.fdrs_to_qvals(np.array(cumulative_decoy_count / cumulative_target_count))
    
    @staticmethod
    def fdrs_to_qvals(fdrs):
        qvals = [0] * len(fdrs)
        if len(fdrs) > 0:
            qvals[len(fdrs)-1] = fdrs[-1]
            for i in range(len(fdrs)-2, -1, -1):
                qvals[i] = min(qvals[i+1], fdrs[i])
        return qvals
    
    def _reorder_columns_for_percolator(self):
        all_columns = self.metrics_val.columns
        first_columns = ['SpecId', 'Label', 'ScanNr']
        last_columns = ['Peptide', 'Protein']
        mid_columns = list(set(all_columns) - set(first_columns) - set(last_columns))
        new_columns = first_columns + sorted(mid_columns) + last_columns
        self.metrics_val = self.metrics_val[new_columns]
    
    def calc(self):
        """
        Adds percolator metadata and feature columns to metrics_val based on PSM metadata
        """
        self.add_common_features()

        self.target_decoy_labels = self.metadata['REVERSE'].apply(Percolator.get_target_decoy_label).to_numpy()

        np.random.seed(1)
        # add Prosit or Andromeda features
        if self.input_type == "prosit":
            fragments_ratio = fr.FragmentsRatio(self.pred_intensities, self.true_intensities)
            fragments_ratio.calc()

            similarity = sim.SimilarityMetrics(self.pred_intensities, self.true_intensities)
            similarity.calc(self.all_features_flag)

            self.metrics_val = pd.concat([self.metrics_val, fragments_ratio.metrics_val, similarity.metrics_val], axis=1)

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
                sampled_idxs = Percolator.sample_balanced_over_bins(
                    self.metadata[['RETENTION_TIME', 'PREDICTED_IRT']])
            else:
                sampled_idxs = Percolator.sample_balanced_over_bins(
                    self.metadata[['RETENTION_TIME', 'PREDICTED_IRT']].iloc[idxs_below_lda_fdr, :])

            aligned_predicted_rts = Percolator.get_aligned_predicted_retention_times(
                self.metadata['RETENTION_TIME'][sampled_idxs],
                self.metadata['PREDICTED_IRT'][sampled_idxs],
                self.metadata['PREDICTED_IRT'])

            self.metrics_val['RT'] = self.metadata['RETENTION_TIME']
            self.metrics_val['pred_RT'] = self.metadata['PREDICTED_IRT']
            self.metrics_val['iRT'] = aligned_predicted_rts
            self.metrics_val['collision_energy_aligned'] = self.metadata['COLLISION_ENERGY']/100.0
            self.metrics_val['abs_rt_diff'] = np.abs(self.metadata['RETENTION_TIME'] - aligned_predicted_rts)
            
            median_abs_error_lda_targets = np.median(self.metrics_val['abs_rt_diff'].iloc[idxs_below_lda_fdr])
            logger.info(f"Median absolute error predicted vs observed retention time on targets < 1% FDR: {median_abs_error_lda_targets}")
            logger.debug(self.metrics_val[['RT', 'pred_RT', 'abs_rt_diff', 'lda_scores']].iloc[idxs_below_lda_fdr, :])
        else:
            self.metrics_val['andromeda'] = self.metadata['SCORE']

        self.add_percolator_metadata_columns()
        if self.input_type == 'prosit':
            # TODO: only add this feature if they are not all zero
            #self.metrics_val['spectral_angle_delta_score'] = Percolator.get_delta_score(self.metrics_val[['ScanNr', 'spectral_angle']], 'spectral_angle')
            pass
        else:
            self.metrics_val['andromeda_delta_score'] = Percolator.get_delta_score(self.metrics_val[['ScanNr', 'andromeda']], 'andromeda')
        
        self._reorder_columns_for_percolator()
