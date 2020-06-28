# README #

### What is this repository for? ###

Set of new Python scripts based on VMTK to automatically (or as automatically 
as possible) extract high quality intracranial aneurysms and vessels netwrok
surface for CFD simulations using the Vacular Modeling Toolkit framework 
(http://www.vmtk.org/).

Aneurysms

Also includes Python library of functions to calculate morphological and 
hemodynamic parameters related to aneurysms geometry and hemodynamics using
ParaView filters and VMTK.

The library works with the paraview.simple module of ParaView. 

### How do I get set up? ###

Use VMTK's Python library.
You need to have a working installation of VMTK in your computer (I recommend
using the Python's Anaconda distribution, then create a new environment 
with VMTK). Once done, include the scripts inside the "newScripts" directory
in the site-packages of the lib dir in the environment location (typically
"env_dir/lib/python3.x/site-packages/vmtk/").
Do not forget to also include the script's name in the __all__ list of the
"vmtkscripts.py" module.

### Contribution guidelines ###


### Who do I talk to? ###

Iago Lessa de Oliveira
UNESP
Ilha Solteira - SP
Brazil
