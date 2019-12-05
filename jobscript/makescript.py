#!/usr/bin/env python3
import subprocess
import glob
import os

with open('jobscript.sh', 'r') as f:
    script = f.read()

scripts = []
for k in glob.glob(os.path.expanduser('~/Documents/umap_mvn2/dataset_all/*')):
    # for i in [0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.5, 0.7, 1.0]: # c
    for i in [1.0]:
        # for j in [9999]:
        # for j in range(0, 19, 6):  # layer
        for j in [19]:
            # for l in [55]:
            for l in [6, 5, 15, 32, 85, 55, 71, 16, 78, 69]:  # seed
                # for m in [0.000]:
                # for m in [0.85, 0.9, 0.95, 0.99, 0.999]:
                for m in [0.999]:
                    # for mul_sig in [1, 2, 3, 4, 5]:
                    for mul_sig in [0]:
                        scriptname = 'josbscript'+str(os.path.basename(k))+'_'+str(i)+'_'+str(j)+'_'+str(l)+'_'+str(m)+'_'+str(mul_sig)+'.sh'
                        scripts.append(scriptname)
                        print(scriptname)
                        with open(scriptname, 'w') as f:
                            f.write(script.format(str(i), str(j), str(k), str(os.path.basename(k)), str(l), str(m), str(mul_sig)))

print(scripts)
if (input('continue ? >')[0] == 'y'):
    for s in scripts:
        cmd = "qsub -g gaa50088 ./{}".format(s)
        subprocess.call(cmd.split())
