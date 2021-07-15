import pytest
import numpy as np
import pandas as pd

import fundamentals.metrics.percolator as perc
import fundamentals.constants as constants


class TestFdrs:
    def test_calculate_fdrs(self):
        sorted_labels = np.array([1, 1, 0, 0])
        np.testing.assert_almost_equal(perc.Percolator.calculate_fdrs(sorted_labels), [0.5, 0.3333333, 0.66666667, 1.0])
    
    def test_get_indices_below_fdr_none(self):
        percolator = perc.Percolator(pd.DataFrame(), None, None)
        percolator.metrics_val['Score'] = [0, 3, 2, 1]
        percolator.target_decoy_labels = [0, 0, 1, 0]
        '''
        idx Score  Label  fdr
        1      3      0  2.0
        2      2      1  1.0
        3      1      0  1.5
        0      0      0  2.0
        '''
        np.testing.assert_equal(percolator.get_indices_below_fdr('Score', fdr_cutoff = 0.4), np.array([]))
        
    def test_get_indices_below_fdr(self):
        percolator = perc.Percolator(pd.DataFrame(), None, None)
        percolator.metrics_val['Score'] = [0, 3, 2, 1]
        percolator.target_decoy_labels = [0, 1, 1, 0]
        '''
        idx Score Label       fdr
        1      3      1  0.500000
        2      2      1  0.333333
        3      1      0  0.666667
        0      0      0  1.000000
        '''
        np.testing.assert_equal(percolator.get_indices_below_fdr('Score', fdr_cutoff = 0.4), np.array([1, 2]))
    
    def test_get_indices_below_fdr_filter_decoy(self):
        percolator = perc.Percolator(pd.DataFrame(), None, None)
        percolator.metrics_val['Score'] = [0, 3, 2, 1, 4, 5, 6, 7]
        percolator.target_decoy_labels = [0, 1, 1, 0, 0, 1, 1, 1]
        '''
        idx  Score Label      fdr
        7      7      1  0.500000
        6      6      1  0.333333
        5      5      1  0.250000
        4      4      0  0.500000
        1      3      1  0.400000
        2      2      1  0.333333
        3      1      0  0.500000
        0      0      0  0.666667
        '''
        np.testing.assert_equal(percolator.get_indices_below_fdr('Score', fdr_cutoff = 0.4), np.array([1, 2, 5, 6, 7]))

class TestLda:
    def test_apply_lda_and_get_indices_below_fdr(self):
        """
        Score_2 adds more discriminative power between targets and decoys
        """
        percolator = perc.Percolator(pd.DataFrame(), None, None)
        percolator.metrics_val['Score'] =         [0.0, 3.0, 2.0, 1.0, 4.0, 5.0, 6.0, 7.0]
        percolator.metrics_val['Score_2'] =       [1.0, 1.5, 2.0, 1.5, 1.0, 1.5, 2.0, 1.5]
        percolator.target_decoy_labels = np.array([0,   1,   1,   0,   0,   1,   1,   1  ])
        '''
        idx lda_scores  Label       fdr
        6    8.396540      1  0.500000
        7    4.967968      1  0.333333
        2    4.396540      1  0.250000
        5    2.967968      1  0.200000
        1    0.967968      1  0.166667
        3   -1.032032      0  0.333333
        4   -2.460603      0  0.500000
        0   -6.460603      0  0.666667
        '''
        np.testing.assert_equal(percolator.apply_lda_and_get_indices_below_fdr(initial_scoring_feature = 'Score', fdr_cutoff = 0.4), np.array([1, 2, 5, 6, 7]))

        
class TestRetentionTimeAlignment:
    
    def test_get_aligned_predicted_retention_times_linear(self):
        observed_rts =  np.linspace(0, 10, 10)*2
        predicted_rts = np.linspace(1, 11, 10)
        predicted_rts_all = np.array([1.5, 2.5, 3.5, 4.5]) # observed = 2*(predicted - 1)
        np.testing.assert_almost_equal(perc.Percolator.get_aligned_predicted_retention_times(observed_rts, predicted_rts, predicted_rts_all), [1, 3, 5, 7], decimal = 3)
    
    def test_get_aligned_predicted_retention_times_noise(self):
        observed_rts =  np.linspace(0, 10, 10)*2 + 0.001*np.random.random(10)
        predicted_rts = np.linspace(1, 11, 10)
        predicted_rts_all = np.array([1.5, 2.5, 3.5, 4.5]) # observed = (predicted - 1)^2
        np.testing.assert_almost_equal(perc.Percolator.get_aligned_predicted_retention_times(observed_rts, predicted_rts, predicted_rts_all), [1, 3, 5, 7], decimal = 3)

    def test_sample_balanced_over_bins(self):
        observed_rts = np.linspace(0, 10, 10) * 2 + 0.001 * np.random.random(10)
        predicted_rts = np.linspace(1, 11, 10)
        retention_time_df = pd.DataFrame()
        retention_time_df['RETENTION_TIME'] = observed_rts
        retention_time_df['PREDICTED_RETENTION_TIME'] = predicted_rts
        sampled_index = perc.Percolator.sample_balanced_over_bins(retention_time_df, sample_size=3)
        np.testing.assert_equal(len(sampled_index), 3)
        np.testing.assert_equal(len(set(sampled_index)), 3)




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
    
    def test_calc(self):
        cols = ['RAW_FILE', 'SCAN_NUMBER', 'MODIFIED_SEQUENCE', 'SEQUENCE', 'CHARGE', 'MASS', 'CALCULATED_MASS', 'SCORE', 'REVERSE', 'FRAGMENTATION', 'MASS_ANALYZER', 'RETENTION_TIME', 'PREDICTED_RETENTION_TIME']
        perc_input = pd.DataFrame(columns=cols)
        perc_input = perc_input.append(
            pd.Series('20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02,7978,AAIGEATRL,AAIGEATRL,2,900.50345678,900.50288029264,60.43600000000001,False,HCD,FTMS,0.5,0.5'.split(','),
                      index=cols), ignore_index=True)
        perc_input = perc_input.append(
            pd.Series('20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02,12304,AAVPRAAFL,AAVPRAAFL,2,914.53379,914.53379,34.006,True,HCD,FTMS,1,1.5'.split(','),
                      index=cols), ignore_index=True)
        perc_input = perc_input.append(
            pd.Series('20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02,12398,AAYFGVYDTAK,AAYFGVYDTAK,2,1204.5764,1204.5764,39.97399999999999,True,HCD,FTMS,1.5,2.5'.split(','),
                      index=cols), ignore_index=True)
        perc_input = perc_input.append(
            pd.Series('20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02,11716,AAYYHPSYL,AAYYHPSYL,2,1083.5025,1083.5025,99.919,False,HCD,FTMS,2,3.5'.split(','),
                      index=cols), ignore_index=True)
        perc_input = perc_input.append(
            pd.Series('20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02,5174,AEDLNTRVA,AEDLNTRVA,2,987.49852,987.49852,87.802,False,HCD,FTMS,2.5,4.5'.split(','),
                      index=cols), ignore_index=True)
        perc_input = perc_input.append(
            pd.Series('20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02,5174,AEDLNTRVA,AEDLNTRVA,2,987.49852,987.49852,62.802,False,HCD,FTMS,3,5.5'.split(','),
                      index=cols), ignore_index=True)
        perc_input = perc_input.append(
            pd.Series('20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02,5174,AEDLNTRVA,AEDLNTRVA,2,987.49852,987.49852,79.802,False,HCD,FTMS,3.5,6.5'.split(','),
                      index=cols), ignore_index=True)
        perc_input = perc_input.append(
            pd.Series('20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02,5174,AEDLNTRVA,AEDLNTRVA,2,987.49852,987.49852,79.802,False,HCD,FTMS,4.0,7.5'.split(','),
                      index=cols), ignore_index=True)
        
                      
        perc_input['SCAN_NUMBER'] = perc_input['SCAN_NUMBER'].astype(int)
        perc_input['CHARGE'] = perc_input['CHARGE'].astype(int)
        perc_input['MASS'] = perc_input['MASS'].astype(float)
        perc_input['CALCULATED_MASS'] = perc_input['CALCULATED_MASS'].astype(float)
        perc_input['REVERSE'] = perc_input['REVERSE'] == "True"
        
        # we need to add noise to the retention times to prevent 0 residuals in the lowess regression
        perc_input['RETENTION_TIME'] = perc_input['RETENTION_TIME'].astype(float) + 1e-7*np.random.random(len(perc_input['RETENTION_TIME']))
        perc_input['PREDICTED_RETENTION_TIME'] = perc_input['PREDICTED_RETENTION_TIME'].astype(float) + 1e-7*np.random.random(len(perc_input['RETENTION_TIME']))
        
        z = constants.EPSILON
        #                                         y1.1  y1.2  y1.3  b1.1  b1.2  b1.3  y2.1  y2.2  y2.3
        predicted_intensities_target = get_padded_array([ 7.2,  2.3, 0.01, 0.02,  6.1,  3.1,    z,    z,    0])
        observed_intensities_target =  get_padded_array([10.2,    z,  1.3,    z,  8.2,    z,  3.2,    z,    0])
        
        predicted_intensities_decoy = get_padded_array([  z, 3.0, 4.0,   z])
        observed_intensities_decoy = get_padded_array([  z,   z, 3.0, 4.0])
        
        predicted_intensities = np.tile(predicted_intensities_target, (len(perc_input), 1))
        observed_intensities = np.tile(observed_intensities_target, (len(perc_input), 1))
        
        predicted_intensities[1, :] = predicted_intensities_decoy
        predicted_intensities[2, :] = predicted_intensities_decoy
        observed_intensities[1, :] = observed_intensities_decoy
        observed_intensities[2, :] = observed_intensities_decoy
        
        percolator = perc.Percolator(perc_input, predicted_intensities, observed_intensities)
        percolator.fdr_cutoff = 0.4
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
        
        np.testing.assert_almost_equal(percolator.metrics_val['RT'][0], 0.5, decimal = 3)
        np.testing.assert_almost_equal(percolator.metrics_val['pred_RT'][0], 0.5, decimal = 3)
        np.testing.assert_almost_equal(percolator.metrics_val['abs_rt_diff'][0], 0.0, decimal = 3)
        
        # check label of second PSM (decoy)
        np.testing.assert_equal(percolator.metrics_val['Label'][1], 0)
        
        # check lowess fit of second PSM
        np.testing.assert_almost_equal(percolator.metrics_val['abs_rt_diff'][1], 0.0, decimal = 3)
        # TODO: figure out why this test fails
        #np.testing.assert_almost_equal(percolator.metrics_val['abs_rt_diff'][2], 0.0, decimal = 3)

def get_padded_array(l, padding_value = 0):
    return np.array([np.pad(l, (0, constants.NUM_IONS - len(l)), 'constant', constant_values=padding_value)])
