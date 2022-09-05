# Observing-Flow-with-He-II
repository for code and link to data
Instructions to running the code:

Before running the code, you will need these python package installed on your computer:
NumPy, Matplotlib, spe2py, spe_loader, getopt, SciPy, datetime, dateutil, photutils, ipympl, sklearn, itertools, importlib and ffmpeg.

To run the code:
1. Create a folder name 'input' in the main directory. Leave turublence.py and turbulence_github.jpynb in the main directory.
2. Copy the background file to 'input' folder. Copy the file you want to study to the 'input' folder.
3. Open turbulence_github.jpynb from Jupyter notebook.
4. In cell 2, copy and paste the background name to the 1s line after 'input/'.
5. In cell 2, copy and paste the name of the file you intend to exam to the 2nd line after 'input/'.
6. Then you can run the cells one by one starting from the beginning.

After execution of the Jupyter Notebook:
(1) FileName_cluster_results.txt contains a list of the centroids for all clusters.

(2) In the correlator_frames subdirectory a file of the displacement vectors for a pair of cluster-centroids in adjacent frames within a chosen Dlimit are listed in a file called CorrelatorResults....txt. 

The example subdirectory contains the codes.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3968838.svg)](https://doi.org/10.5281/zenodo.7051680)
