import pytest
import numpy as np
import pandas as pd

import fundamentals.metrics.percolator as perc
import fundamentals.constants as constants


class TestPercolator:
    def test_get_scannr(self):
        np.testing.assert_equal(perc.Percolator.get_scannr(('20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02', 7978)), 171184297275363)
    
    def test_get_specid(self):
        np.testing.assert_string_equal(perc.Percolator.get_specid(('rawfile', 1234, 'ABCD', 2)), 'rawfile-1234-ABCD-2')
    
    def test_count_missed_cleavages(self):
        np.testing.assert_equal(perc.Percolator.count_missed_cleavages('AKAAAAKAK'), 2)
    
    def test_count_arginines_and_lysines(self):
        np.testing.assert_equal(perc.Percolator.count_arginines_and_lysines('ARAAAAKAK'), 3)
    
    def test_calculate_mass_difference(self):
        np.testing.assert_almost_equal(perc.Percolator.calculate_mass_difference((1000.0, 1001.2)), 1.2)
    
    def test_calculate_mass_difference_ppm(self):
        np.testing.assert_almost_equal(perc.Percolator.calculate_mass_difference_ppm((1000.0, 1001.2)), 1200.0)
    
    def test_get_target_decoy_label_target(self):
        reverse = False
        np.testing.assert_equal(perc.Percolator.get_target_decoy_label(reverse), 1)
    
    def test_get_target_decoy_label_decoy(self):
        reverse = True
        np.testing.assert_equal(perc.Percolator.get_target_decoy_label(reverse), 0)
    
    def test_get_aligned_predicted_retention_times_linear(self):
        predicted_rts = [1, 2, 3, 4]
        observed_rts = [2, 2.5, 3, 3.5]
        np.testing.assert_almost_equal(perc.Percolator.get_aligned_predicted_retention_times(predicted_rts, observed_rts), [2, 2.5, 3, 3.5])
    
    def test_get_aligned_predicted_retention_times_squared(self):
        predicted_rts = [1, 2, 3, 4]
        observed_rts = [1, 4, 9, 16]
        np.testing.assert_almost_equal(perc.Percolator.get_aligned_predicted_retention_times(predicted_rts, observed_rts), [1, 4, 9, 16])
    
    
    def test_calc(self):
        cols = ['RAW_FILE', 'SCAN_NUMBER', 'MODIFIED_SEQUENCE', 'SEQUENCE', 'CHARGE', 'MASS', 'CALCULATED_MASS', 'SCORE', 'REVERSE', 'FRAGMENTATION', 'MASS_ANALYZER', 'RETENTION_TIME', 'PREDICTED_RETENTION_TIME']
        perc_input = pd.DataFrame(columns=cols)
        perc_input = perc_input.append(
            pd.Series('20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02,7978,AAIGEATRL,AAIGEATRL,2,900.50345678,900.50288029264,60.43600000000001,False,HCD,FTMS,0.5,0.5'.split(','),
                      index=cols), ignore_index=True)
        perc_input = perc_input.append(
            pd.Series('20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02,12304,AAVPRAAFL,AAVPRAAFL,2,914.53379,914.53379,94.006,True,HCD,FTMS,1,1.5'.split(','),
                      index=cols), ignore_index=True)
        perc_input = perc_input.append(
            pd.Series('20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02,12398,AAYFGVYDTAK,AAYFGVYDTAK,2,1204.5764,1204.5764,79.97399999999999,False,HCD,FTMS,1.5,2.5'.split(','),
                      index=cols), ignore_index=True)
        perc_input = perc_input.append(
            pd.Series('20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02,11716,AAYYHPSYL,AAYYHPSYL,2,1083.5025,1083.5025,99.919,False,HCD,FTMS,2,3.5'.split(','),
                      index=cols), ignore_index=True)
        perc_input = perc_input.append(
            pd.Series('20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02,5174,AEDLNTRVA,AEDLNTRVA,2,987.49852,987.49852,99.802,False,HCD,FTMS,2.5,4.5'.split(','),
                      index=cols), ignore_index=True)
        perc_input['SCAN_NUMBER'] = perc_input['SCAN_NUMBER'].astype(int)
        perc_input['CHARGE'] = perc_input['CHARGE'].astype(int)
        perc_input['MASS'] = perc_input['MASS'].astype(float)
        perc_input['CALCULATED_MASS'] = perc_input['CALCULATED_MASS'].astype(float)
        perc_input['REVERSE'] = perc_input['REVERSE'] == "True"
        perc_input['RETENTION_TIME'] = perc_input['RETENTION_TIME'].astype(float)
        perc_input['PREDICTED_RETENTION_TIME'] = perc_input['PREDICTED_RETENTION_TIME'].astype(float)
        
        predicted_intensities = [[]]
        observed_intensities = [[]]
        percolator = perc.Percolator(perc_input, predicted_intensities, observed_intensities)
        percolator.calc()

        # meta data for percolator
        np.testing.assert_string_equal(percolator.metrics_val['SpecId'][0], '20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02-7978-AAIGEATRL-2')
        np.testing.assert_equal(percolator.metrics_val['Label'][0], 1)
        np.testing.assert_equal(percolator.metrics_val['ScanNr'][0], 171184297275363)
        np.testing.assert_almost_equal(percolator.metrics_val['ExpMass'][0], 900.50345678)
        np.testing.assert_string_equal(percolator.metrics_val['Peptide'][0], 'AAIGEATRL')
        np.testing.assert_string_equal(percolator.metrics_val['Protein'][0], 'AAIGEATRL') # we don't need the protein ID to get PSM / peptide results
        
        # features
        np.testing.assert_equal(percolator.metrics_val['missedCleavages'][0], 1)
        np.testing.assert_equal(percolator.metrics_val['KR'][0], 1)
        np.testing.assert_equal(percolator.metrics_val['sequence_length'][0], 9)
        np.testing.assert_almost_equal(percolator.metrics_val['Mass'][0], 900.50345678) # this is the experimental mass as a feature
        np.testing.assert_almost_equal(percolator.metrics_val['deltaM_Da'][0], -0.0005764873)
        np.testing.assert_almost_equal(percolator.metrics_val['absDeltaM_Da'][0], 0.0005764873)
        np.testing.assert_almost_equal(percolator.metrics_val['deltaM_ppm'][0], -0.64018339472)
        np.testing.assert_almost_equal(percolator.metrics_val['absDeltaM_ppm'][0], 0.64018339472)
        np.testing.assert_equal(percolator.metrics_val['Charge2'][0], 1)
        np.testing.assert_equal(percolator.metrics_val['Charge3'][0], 0)
        np.testing.assert_equal(percolator.metrics_val['UnknownFragmentationMethod'][0], 0)
        np.testing.assert_equal(percolator.metrics_val['HCD'][0], 1)
        np.testing.assert_equal(percolator.metrics_val['CID'][0], 0)
        
        np.testing.assert_almost_equal(percolator.metrics_val['RT'][0], 0.5)
        np.testing.assert_almost_equal(percolator.metrics_val['pred_RT'][0], 0.5)
        np.testing.assert_almost_equal(percolator.metrics_val['abs_rt_diff'][0], 0.0)
        
        # check label of second PSM (decoy)
        np.testing.assert_equal(percolator.metrics_val['Label'][1], 0)
        
        # check lowess fit of second PSM
        np.testing.assert_almost_equal(percolator.metrics_val['abs_rt_diff'][1], 0.0)
