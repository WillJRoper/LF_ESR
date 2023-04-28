import sys
import os
import h5py

import esr.generation.duplicate_checker
import esr.fitting.test_all
import esr.fitting.test_all_Fisher
import esr.fitting.match
import esr.fitting.combine_DL
import esr.fitting.plot

from likelihood import LFLikelihood

# Set up the function generation 
runname = 'ext_maths'
complexity = int(sys.argv[1])
esr.generation.duplicate_checker.main(runname, complexity, seed=42)

# Define the path to the data
datapath = "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/" \
    + "flares.hdf5"

# Get what snapshot we are doing
tags = ['010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000',
        '006_z009p000', '005_z010p000']
snap = tags[int(sys.argv[2])]

# Define the output path
outpath = "FLARES_ESR_DATA/complexity%d/%s" % (complexity, snap)

# Make sure the output directory exists
if not os.path.exists(outpath):
    os.makedirs(outpath)

# Set up the likelihood
likelihood = LFLikelihood(datapath, "FLARES-LF", snap,
                          data_dir=outpath, base_dir=outpath,
                          param_set=runname)
print("Got likelihood")

# Run the fitting
esr.fitting.test_all.main(complexity, likelihood)
esr.fitting.test_all_Fisher.main(complexity, likelihood)
esr.fitting.match.main(complexity, likelihood)
esr.fitting.combine_DL.main(complexity, likelihood)
print("Plotting in", likelihood.fig_dir)
esr.fitting.plot.main(complexity, likelihood)

