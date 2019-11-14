#!/usr/bin/env python3
import subprocess
import glob
import os

with open('jobscript.sh', 'r') as f:
    script = f.read()

scripts = []
for k in glob.glob(os.path.expanduser('~/Documents/umap_mnv2/dataset_all/*')):
    # for i in [0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.5, 0.7, 1.0]: # c
    for i in [1.0]:
        # for j in [12]:
        for j in range(0, 19, 6):  # layer
            # for l in [7, 5, 14, 32, 84, 55, 71, 16, 78]:  # seed
            for l in [55]:
                scriptname = 'josbscript'+str(os.path.basename(k))+'_'+str(i)+'_'+str(j)+'_'+str(l)+'.sh'
                scripts.append(scriptname)
                print(scriptname)
                with open(scriptname, 'w') as f:
                    f.write(script.format(str(i), str(j), str(k), str(os.path.basename(k)), str(l)))

print(scripts)
if (input('continue ? >')[0] == 'y'):
    for s in scripts:
        cmd = "qsub -g gaa50088 ./{}".format(s)
        subprocess.call(cmd.split())
