import os
import sys
import unittest

from lib import cfdtools as cfd

class TestCfdTools(unittest.TestCase):
    # Write test for observed accuracy order
    # # Observed order of accuracy
    # accuracy_order = cfd.observed_accuracy_order(
    #                      f_fine,
    #                      f_intermediate,
    #                      f_coarse,
    #                      r
    #                  )

    def test_RichardsonErroAndGCI(self):
        # Values from Example 19.8 of Roache
        GCI_Roache = 15.2
        relatDiff_Roache = 2.85

        # The value in Roache is 5.07 but is the result of a "middle"
        # approximation
        RichardsonError_Roache = 5.06

        # This refinement ratio is given by h_coarse/h_fine
        r = cfd.refinement_ratio(1600, 2000, dimensionality=1)

        # Theoretical accuracy order assumed by Roache
        accuracy_order = 2.0

        # Solutions with meshes
        f_intermediate = -544.48
        f_fine = -529.41

        # Compute relative differences
        relatDiff = 100.0*abs((f_fine - f_intermediate)/f_fine)

        # Richardson error
        RichardsonError = 100.0*abs(
                                cfd.Richardson_error_estimator(
                                    f_fine,
                                    f_intermediate,
                                    r,
                                    accuracy_order
                                )
                            )

        # GCI of fine mesh
        GCI = 100.0*abs(
                    cfd.GCI_fine_mesh(
                        f_fine,
                        f_intermediate,
                        r,
                        accuracy_order
                    )
                )

        print(
            "Relat. Diff. = {}, Richardson Error = {}, GCI = {}".format(
                relatDiff,
                RichardsonError,
                GCI,
            )
        )

        # Using the modulo operator for string formatting
        # this avoind the use of round function
        self.assertEqual(
            float("%.2f" % relatDiff),
            relatDiff_Roache,
            "Error in relative difference calculation."
        )

        self.assertEqual(
            float("%.2f" % RichardsonError),
            RichardsonError_Roache,
            "Error in Richardson error calculation."
        )

        self.assertEqual(
            float("%.1f" % GCI),
            GCI_Roache,
            "Error in GCI calculation."
        )

if __name__=='__main__':
    unittest.main()
