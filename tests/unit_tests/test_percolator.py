import pytest
import numpy as np
import pandas as pd

import fundamentals.metrics.percolator as perc
import fundamentals.constants as constants


class TestPercolator:

    def test_calc(self):
        cols = ['RAW_FILE', 'SCAN_NUMBER', 'MODIFIED_SEQUENCE', 'CHARGE', 'MASS', 'SCORE', 'REVERSE', 'FRAGMENTATION', 'MASS_ANALYZER']
        perc_input = pd.DataFrame(columns=cols)
        perc_input = perc_input.append(pd.Series(
            '20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02,7978,AAIGEATRL,2,900.5028800000001,60.43600000000001,False,HCD,FTMS'.split(','),
            index=cols), ignore_index=True)
        perc_input = perc_input.append(
            pd.Series('20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02,12304,AAVPRAAFL,2,914.53379,94.006,False,HCD,FTMS'.split(','),
                      index=cols), ignore_index=True)
        perc_input = perc_input.append(pd.Series(
            '20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02,12398,AAYFGVYDTAK,2,1204.5764,79.97399999999999,False,HCD,FTMS'.split(','),
            index=cols), ignore_index=True)
        perc_input = perc_input.append(
            pd.Series('20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02,11716,AAYYHPSYL,2,1083.5025,99.919,False,HCD,FTMS'.split(','),
                      index=cols), ignore_index=True)
        perc_input = perc_input.append(
            pd.Series('20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02,5174,AEDLNTRVA,2,987.49852,99.802,False,HCD,FTMS'.split(','),
                      index=cols), ignore_index=True)

        percolator = perc.Percolator(perc_input, predicted_intensities, observed_intensities)
        percolator.calc()

        # counting metrics
        np.testing.assert_string_equal(percolator.metrics_val['SpecId'][0], '20210122_0263_TMUCLHan_Peiru_DDA_IP_C797S_02-7978-AAIGEATRL-2')
        np.testing.assert_equal(percolator.metrics_val['ScanNr'][0], 171184297275363)
        np.testing.assert_string_equal(percolator.metrics_val['Peptide'][0], 'AAIGEATRL')
        np.testing.assert_equal(percolator.metrics_val['missedCleavages'][0], 1)
        np.testing.assert_equal(percolator.metrics_val['sequence_length'][0], 9)
        np.testing.assert_almost_equal(percolator.metrics_val['Mass'][0], 900.50288)
        np.testing.assert_string_equal(percolator.metrics_val['ExpMass'][0], 900.50288)
        np.testing.assert_string_equal(percolator.metrics_val['Label'][0], 1)
        np.testing.assert_string_equal(percolator.metrics_val['Protein'][0], 'AAIGEATRL')
        np.testing.assert_string_equal(percolator.metrics_val['deltaM_ppm'][0], '') # 900.492



