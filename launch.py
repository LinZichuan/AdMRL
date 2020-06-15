import numpy as np
import os, sys
from subprocess import Popen
import subprocess
from rllab.misc.instrument import VariantGenerator, variant
class VG(VariantGenerator):
    @variant
    def initnslbo(self): return [20] #[20, 30, 40]
    @variant
    def nslbo(self): return [3] #3
    @variant
    def warmniter(self): return [40] #10
    @variant
    def slboniter(self): return [20] #10
    @variant
    def piter(self): return [20]  #20
    @variant
    def miter(self): return [100]
    @variant
    def atype(self): return ['gae']
    @variant
    def inittask(self): return ['none']
    @variant
    def seed(self): return [1]
    @variant
    def alpha(self):
        return [1.0]
    @variant
    def adv(self):
        return [1]  #0:Uniform, 1:AdMRL, 2:Gaussian
    @variant
    def nsample(self):
        return [10000]
    @variant
    def taskname(self):
        return ['Ant2D']

skip_list = ['inittask']

variants = VG().variants()
idx = 0
for i,v in enumerate(variants):
    if not os.path.exists('logs'): os.makedirs('logs')
    setting = "result"
    for k in v.keys():
        if 'hidden_keys' in k: continue
        if k in skip_list: continue
        setting += f"_{k}{v[k]}"
    print (setting)
    cmd = ["python", "main.py"]
    cmd.append(f'--setting={setting}')
    for k in v.keys():
        if 'hidden_keys' in k: continue
        cmd.append(f"--{k}={v[k]}")
    e = os.environ.copy()
    e['CUDA_VISIBLE_DEVICES'] = f'{idx%4}'
    idx += 1
    print (cmd)
    print (' '.join(cmd))
    with open(f"logs/{setting}.txt", 'w') as out:
        Popen(cmd, env=e, stdout=out, stderr=out, stdin=out)
