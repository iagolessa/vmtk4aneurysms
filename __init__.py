"""vmtk4aneurysms -- tools to study vascular models with aneurysms.

vmtk4aneurysms is a collection of modules providing tools to manipulate and
morphologically characterize vascular models, with aneurysms or not (although
the main initial purpose to develop this library was to study intracranial
aneurysms). The main modules provide representations for vascular models and
also an aneurysms sac.

The library also provides tools to compute and manipulate fields defined on the
surfaces, mainly hemodynamics related to blood flow in the vascular models and
wall mechanical variables.  The conversion tools accepts conversion from
OpenFOAM simulations.

It also accounts with a set of VMTK-like scripts using the Pypes module that
may be useful, although some of them I developed for more specialized
operations related to intracranial aneurysms. The same comment apply to the
'apps' folder which accounts with Python scripts for pre- and post-processing
of numerical simulations in OpenFOAM.

The whole library is heavily dependent on the Visualization Toolkit (VTK), the
Vascular Modeling Toolkit (VMTK). It also depends on MorphMan and other general
Python libraries such as SciPy, NumPy, and Pandas.

You need to have a working installation of VMTK in your computer. I recommend
using the Python's Anaconda distribution, then create a new environment with
VMTK. The MorphMan library can be installed in this environment then.

Iago Lessa de Oliveira
iago.oliveira@unesp.br
"""
