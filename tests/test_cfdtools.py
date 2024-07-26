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

    # def test_RichardsonErrorBenchmarkSolution(self):
    #      # ** Test of the computation of p with regular r **

    #     # Testing with benchmark solution by de Vahl Davis papers with the
    #     # cavity problem Build dictionary with solutions given by de Vahl Davis
    #     verVelocityCavity = {
    #         "Ra_1e3": {
    #             "Coarse": {"h": 0.1,   "v": 3.449},
    #             "Interm": {"h": 0.05,  "v": 3.629},
    #             "Finest": {"h": 0.025, "v": 3.679}
    #         },
    #         "Ra_1e4": {
    #             "Coarse": {"h": 0.1,   "v": 18.055},
    #             "Interm": {"h": 0.05,  "v": 19.197},
    #             "Finest": {"h": 0.025, "v": 19.509}
    #         },
    #         "Ra_1e5": {
    #             "Coarse": {"h": 0.025,  "v": 66.73},
    #             "Interm": {"h": 0.016,  "v": 67.91},
    #             "Finest": {"h": 0.0125, "v": 68.22}
    #         },
    #         "Ra_1e6": {
    #             "Coarse": {"h": 0.025,  "v": 206.32},
    #             "Interm": {"h": 0.016,  "v": 214.64},
    #             "Finest": {"h": 0.0125, "v": 216.75}
    #         }
    #     }

    #     # Same refinement ratio in this case
    #     Ra = "Ra_1e3"

    #     # Get solution
    #     v3 = verVelocityCavity[Ra]["Coarse"]["v"]
    #     v2 = verVelocityCavity[Ra]["Interm"]["v"]
    #     v1 = verVelocityCavity[Ra]["Finest"]["v"]

    #     # Get mesh sizes
    #     h3 = verVelocityCavity[Ra]["Coarse"]["h"]
    #     h2 = verVelocityCavity[Ra]["Interm"]["h"]
    #     h1 = verVelocityCavity[Ra]["Finest"]["h"]

    #     r23 = h3/h2
    #     r12 = h2/h1

    #     self.assertEqual(r12 == r23)

    #     # Use this version
    #     cfd.observed_accuracy_order(v1,v2,v3,r12,r23)

    #     # Now lets test a vectorized version.
    #     # We assume that the Ra number is a field
    #     # So each mesh has a field of velocity
    #     Ras = ['Ra_1e6', 'Ra_1e3', 'Ra_1e5', 'Ra_1e4']

    #     vMesh1 = np.array([verVelocityCavity[Ra]["Finest"]["v"]
    #                        for Ra in Ras])

    #     vMesh2 = np.array([verVelocityCavity[Ra]["Interm"]["v"]
    #                        for Ra in Ras])

    #     vMesh3 = np.array([verVelocityCavity[Ra]["Coarse"]["v"]
    #                        for Ra in Ras])

    #     # Use this version
    #     pField = cfd.observed_accuracy_order(vMesh1,vMesh2,vMesh3,r12,r23)
    #     100.0*cfd.Richardson_error_estimator(vMesh1, vMesh2, r12, pField)
    #     100.0*cfd.GCI_fine_mesh(vMesh2, vMesh3, r23, pField)

if __name__=='__main__':
    unittest.main()
