optional: conda activate fenicsx-env

run the coupled system: (same for ChannelFlowNoObstacle.py)
adjust parameters in ChannelFlowCoupledSystem.py
run ChannelFlowCoupledSystem.py by running "python3 ChannelFlowCoupledSystem.py"

in order to run FENE-P fokker-planck pipeline:
uncomment line 299 in file models/FENE-P/fene_p.py
adjust parameters in pipline()-function of file models/FENE-P/fene_p.py
run models/FENE-P/fene_p.py by running "python3 models/FENE-P/fene_p.py"

create plots:
use loading functions to load experiments of desire
(un)comment marked paths to define what needs to be plotted
run resulty_post_processing.py

IMPORTANT: check that "plots/experiments" and "results/arrays/experiments" folders are available in the root folder of
the project for the pipeline to create subdirectories for different experiments