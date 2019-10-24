#!/usr/bin/env python3
import subprocess

with open('jobscript.sh', 'r') as f:
    script = f.read()

scripts = []
# for i in [0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.5, 0.7, 1.0]:
for i in [0.01, 0.1, 1.0]:
    for j in range(0, 19, 4):
        scriptname = 'josbscript'+str(i)+'_'+str(j)+'.sh'
        scripts.append(scriptname)
        print(scriptname)
        with open(scriptname, 'w') as f:
            f.write(script.format(str(i), str(j)))

print(scripts)
if (input('continue ? >')[0] == 'y'):
    for s in scripts:
        cmd = "qsub -g gaa50088 ./{}".format(s)
        subprocess.call(cmd.split())
