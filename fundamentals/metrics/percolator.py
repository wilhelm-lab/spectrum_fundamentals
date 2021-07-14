import hashlib

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from .metric import Metric


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
    
    def __init__(self, metadata, pred_intensities, true_intensities):
        self.metadata = metadata
        super().__init__(pred_intensities, true_intensities)
    
    @staticmethod    
    def get_aligned_predicted_retention_times(predicted_retention_times, observed_retention_times, xvals = []):
        """
        Use IRT value here
        """
        if len(xvals) > 0:
            aligned_rts = lowess(observed_retention_times, predicted_retention_times, xvals = xvals)
        else:
            aligned_rts = lowess(observed_retention_times, predicted_retention_times)
        print(aligned_rts)
        aligned_rts_observed, aligned_rts_predicted = zip(*aligned_rts)
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
    def get_specid(metadata_subset):
        """
        Create a unique identifier used as spectrum id in percolator, this is not parsed by percolator but functions as a key to map percolator results back to our internal representation
        :param metadata_subset: tuple of (raw_file, scan_number, modified_sequence, charge)
        :return: percolator spectrum id
        """
        raw_file, scan_number, modified_sequence, charge = metadata_subset
        s = "{}-{}-{}-{}".format(
            raw_file, scan_number, modified_sequence, charge
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
        :return: target/decoy label for percolator, 1 = Target, 0 = Decoy
        """
        return 0 if reverse else 1

    def calc(self):
        """
        Adds percolator metadata and feature columns to metrics_val based on PSM metadata
        """
        # PSM metadata
        self.metrics_val['SpecId'] = self.metadata[['RAW_FILE', 'SCAN_NUMBER', 'MODIFIED_SEQUENCE', 'CHARGE']].apply(Percolator.get_specid, axis=1)
        self.metrics_val['Label'] = self.metadata['REVERSE'].apply(Percolator.get_target_decoy_label)
        self.metrics_val['ScanNr'] = self.metadata[['RAW_FILE', 'SCAN_NUMBER']].apply(Percolator.get_scannr, axis = 1)
        self.metrics_val['ExpMass'] = self.metadata['MASS']
        self.metrics_val['Peptide'] = self.metadata['MODIFIED_SEQUENCE']
        self.metrics_val['Protein'] = self.metadata['MODIFIED_SEQUENCE'] # we don't need the protein ID to get PSM / peptide results, fill with peptide sequence
        
        # PSM features
        self.metrics_val['missedCleavages'] = self.metadata['SEQUENCE'].apply(Percolator.count_missed_cleavages)
        self.metrics_val['KR'] = self.metadata['SEQUENCE'].apply(Percolator.count_arginines_and_lysines)
        self.metrics_val['sequence_length'] = self.metadata['SEQUENCE'].apply(lambda x : len(x))
        
        self.metrics_val['Mass'] = self.metadata['MASS'] # this is the experimental mass used as a feature
        self.metrics_val['deltaM_Da'] = self.metadata[['MASS', 'CALCULATED_MASS']].apply(Percolator.calculate_mass_difference, axis = 1)
        self.metrics_val['absDeltaM_Da'] = np.abs(self.metrics_val['deltaM_Da'])
        self.metrics_val['deltaM_ppm'] = self.metadata[['MASS', 'CALCULATED_MASS']].apply(Percolator.calculate_mass_difference_ppm, axis = 1)
        self.metrics_val['absDeltaM_ppm'] = np.abs(self.metrics_val['deltaM_ppm'])
        
        self.metrics_val['Charge2'] = (self.metadata['CHARGE'] == 2).astype(int)
        self.metrics_val['Charge3'] = (self.metadata['CHARGE'] == 3).astype(int)
        
        self.metrics_val['UnknownFragmentationMethod'] = (~self.metadata['FRAGMENTATION'].isin(['HCD', 'CID'])).astype(int)
        self.metrics_val['HCD'] = (self.metadata['FRAGMENTATION'] == 'HCD').astype(int)
        self.metrics_val['CID'] = (self.metadata['FRAGMENTATION'] == 'CID').astype(int)
        
        self.metrics_val['RT'] = self.metadata['RETENTION_TIME']
        self.metrics_val['pred_RT'] = self.metadata['PREDICTED_RETENTION_TIME']
        aligned_predicted_rts = Percolator.get_aligned_predicted_retention_times(self.metadata['PREDICTED_RETENTION_TIME'], self.metadata['RETENTION_TIME'])
        print(aligned_predicted_rts)
        self.metrics_val['abs_rt_diff'] = np.abs(self.metadata['RETENTION_TIME'] - aligned_predicted_rts)
        
