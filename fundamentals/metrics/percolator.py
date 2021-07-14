import hashlib

import numpy as np
import pandas as pd

from .metric import Metric


class Percolator(Metric):
    """
    Expects the following metadata columns:
    RAW_FILE
    SCAN_NUMBER
    MODIFIED_SEQUENCE
    CHARGE
    MASS
    CALCULATED_MASS
    SCORE
    REVERSE
    FRAGMENTATION
    MASS_ANALYZER
    """
    metadata: pd.DataFrame
    
    def __init__(self, metadata, pred_intensities, true_intensities):
        self.metadata = metadata
        super().__init__(pred_intensities, true_intensities)
        
    def fit_loess(self):
        """
        Use IRT value here
        """
        

    @staticmethod
    def get_scannr(metadata_subset):
        raw_file, scan_number = metadata_subset
        s = "{}{}".format(raw_file, scan_number).encode()
        return int(hashlib.sha224(s).hexdigest()[:12], 16)
    
    @staticmethod
    def get_specid(metadata_subset):
        raw_file, scan_number, modified_sequence, charge = metadata_subset
        s = "{}-{}-{}-{}".format(
            raw_file, scan_number, modified_sequence, charge
        )
        return s
    
    @staticmethod
    def count_missed_cleavages(sequence):
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
        Calculate percolator metadata and feature columns from PSM metadata
        """
        
        # PSM metadata
        self.metrics_val['SpecId'] = self.metadata[['RAW_FILE', 'SCAN_NUMBER', 'MODIFIED_SEQUENCE', 'CHARGE']].apply(Percolator.get_specid, axis=1)
        self.metrics_val['Label'] = self.metadata['REVERSE'].apply(Percolator.get_target_decoy_label)
        self.metrics_val['ScanNr'] = self.metadata[['RAW_FILE', 'SCAN_NUMBER']].apply(Percolator.get_scannr, axis = 1)
        self.metrics_val['ExpMass'] = self.metadata['MASS']
        self.metrics_val['Peptide'] = self.metadata['MODIFIED_SEQUENCE']
        self.metrics_val['Protein'] = self.metadata['MODIFIED_SEQUENCE'] # we don't need the protein ID to get PSM / peptide results, fill with peptide sequence
        
        # PSM features
        self.metrics_val['missedCleavages'] = self.metadata['MODIFIED_SEQUENCE'].apply(Percolator.count_missed_cleavages)
        self.metrics_val['KR'] = self.metadata['MODIFIED_SEQUENCE'].apply(Percolator.count_arginines_and_lysines)
        self.metrics_val['sequence_length'] = self.metadata['MODIFIED_SEQUENCE'].apply(lambda x : len(x)) # TODO: remove modifications
        
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
        
        self.metrics_val['RT'] = self.metadata['RT']
        self.metrics_val['pred_RT'] = self.metadata['PREDICTED_RT']
        self.metrics_val['abs_rt_diff'] = self.metadata['RT'] - self.metadata['PREDICTED_RT']
        
