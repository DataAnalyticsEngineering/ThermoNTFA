OPTIONS="-q --show-progress"
DATASET=https://darus.uni-stuttgart.de/api/access/datafile/:persistentId\?persistentId\=doi:10.18419/darus-4509
wget $OPTIONS $DATASET/8 -O data/results/NTFA293K_coarse_temp_293-800.h5
wget $OPTIONS $DATASET/7 -O data/results/NTFA293K_fine_temp_293-800.h5
wget $OPTIONS $DATASET/2 -O data/ntfa/ntfa_6loadings_10samples_N12.h5
wget $OPTIONS $DATASET/10 -O data/ntfa/ntfa_6loadings_10samples_N18.h5
wget $OPTIONS $DATASET/4 -O data/ntfa/ntfa_6loadings_10samples_N24.h5
wget $OPTIONS $DATASET/9 -O data/results/reproduction_6loadings_100samples_N12.h5
wget $OPTIONS $DATASET/3 -O data/results/reproduction_6loadings_100samples_N18.h5
wget $OPTIONS $DATASET/1 -O data/results/reproduction_6loadings_100samples_N24.h5
wget $OPTIONS $DATASET/5 -O data/rve/rve_thermoplastic_6loadings_10samples.h5
wget $OPTIONS $DATASET/6 -O data/rve/twoscale_notch_rve.h5
