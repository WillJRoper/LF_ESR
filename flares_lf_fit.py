import sys
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
print("Generated function set...")

# Define the path to the data
datapath = "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/" \
    + "flares.hdf5"

# Get what snapshot we are doing
tags = ['010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000',
        '006_z009p000', '005_z010p000']
snap = tags[int(sys.argv[2])]


# Set up the likelihood
likelihood = LFLikelihood(datapath, "FLARES-LF", snap,
                          data_dir="FLARES_ESR_DATA")


# Run the fitting
esr.fitting.test_all.main(comp, likelihood)
esr.fitting.test_all_Fisher.main(comp, likelihood)
esr.fitting.match.main(comp, likelihood)
esr.fitting.combine_DL.main(comp, likelihood)
# esr.fitting.plot.main(comp, likelihood)

