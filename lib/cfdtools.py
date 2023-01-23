# Copyright (C) 2022, Iago L. de Oliveira

# vmtk4aneurysms is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Collection of tools to perform CFD-related calculations."""

import numpy as np

from numpy.linalg import norm
from typing import Union, Optional

# Define natural logarithm "properly"
ln = np.log

def refinement_ratio(
        n_cells_coarse: int,
        n_cells_fine: int,
        dimensionality: int=3
    )   -> float:
    """Compute the (effective) refinement ratio between two meshes."""

    return (n_cells_fine/n_cells_coarse)**(1/dimensionality)

def observed_accuracy_order(
        f_fine: Union[float, np.ndarray],
        f_intermediate: Union[float, np.ndarray],
        f_coarse: Union[float, np.ndarray],
        r_interm_fine: Union[int, float],
        r_coarse_intermed: Optional[Union[int, float]]=None,
        relax: float=0.3,
        tol: float=1e-6
    )   -> Union[float, np.ndarray]:
    """Given an equal refinement ratio between three meshes, compute the
    observed order of accuracy of the solutions.

    The observed order of accuracy of the given solutions can be estimated by
    using three solution in sistematically-refined meshes (coarse f_3,
    intermediate f_2, and fine f_1) if the refinement ratio is uniform and
    equal to r as:

    .. math::
        p = ln((f_3 - f_2)/(f_2 - f_1))/ln(r)

    In case the refinement ratio is irregular, then the observed rate is the
    solution of a transcendental equation as given by Eq. (19.10.6.3A) of
    Roache's book "Fundamentals of Computational Flui Dynamics" and the
    solution here is obtained by an iterative approach proposed by the author.
    """

    # Check whether any solution is float
    if any([type(f) is float or type(f) is int
            for f in [f_fine, f_intermediate, f_coarse]]):
        f_fine = np.array([f_fine])
        f_intermediate = np.array([f_intermediate])
        f_coarse = np.array([f_coarse])

    # Compute solution differentials (Roache symbols)
    epsilon23 = f_coarse - f_intermediate
    epsilon12 = f_intermediate - f_fine

    if r_coarse_intermed is None:
        # Then r12 == r23
        # i.e. the refinement was regular
        r = r_interm_fine
        p = ln(epsilon23/epsilon12)/ln(r)

    else:
        arrayShape = f_fine.shape

        # Solution valid for non-uniform r
        # Iterative method recomended by Roache
        p = np.full(arrayShape, 1.0)
        old_p = np.full(arrayShape, 0.0)

        # For consistency with Roach symbols
        r23 = r_coarse_intermed
        r12 = r_interm_fine

        # Use iterative method proposed by Roache
        while np.all(norm(old_p - p, ord=np.inf) >= np.full(arrayShape, tol)):
        # for i in range(20):
            old_p = p

            refineRatio = (r12**p - 1.0)/(r23**p - 1.0)
            beta = refineRatio*epsilon23/epsilon12

            p = relax*p + (1.0 - relax)*ln(beta)/ln(r12)

    return p

def Richardson_error_estimator(
        f_fine: Union[float, np.ndarray],
        f_coarse: Union[float, np.ndarray],
        r: Union[int, float],
        p: Union[float, np.ndarray]
    )   -> Union[float, np.ndarray]:
    """Compute the (relative) Richardson error estimator of the finer mesh."""

    # Get relative difference
    relativeDiff = np.abs(f_fine - f_coarse)/f_fine

    return relativeDiff/(r**p - 1.0)

def GCI_fine_mesh(
        f_fine: Union[float, np.ndarray],
        f_coarse: Union[float, np.ndarray],
        r: Union[int, float],
        p: Union[float, np.ndarray],
        Fs=3.0
    )   -> Union[float, np.ndarray]:
    """Compute the (relative) Grid Convergence Index of the finer mesh
    passed."""

    # Get the Richardson error estimator
    RichardsonError = Richardson_error_estimator(f_fine, f_coarse, r, p)

    return Fs*RichardsonError

def GCI_coarse_mesh(
        f_fine: Union[float, np.ndarray],
        f_coarse: Union[float, np.ndarray],
        r: Union[int, float],
        p: Union[float, np.ndarray],
        Fs=3.0
    )   -> Union[float, np.ndarray]:
    """Compute the (relative) Grid Convergence Index of the coarse mesh
    passed."""

    # Get the Richardson error estimator
    gciFineMesh = GCI_fine_mesh(
                        f_fine,
                        f_coarse,
                        r,
                        p,
                        Fs=Fs
                    )

    return (r**p)*gciFineMesh
