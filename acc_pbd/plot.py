import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Acc stable constrainted dyanmics')
parser.add_argument('--output', '-o', nargs=1, default="image/comparision.png", type=str, required=False)
parser.add_argument('--input1', '-i1',nargs=1, default="data/global.txt", type=str, required=False)
parser.add_argument('--input2', '-i2',nargs=1, default="data/local.txt", type=str, required=False)

args = parser.parse_args()
output_file = args.output if type(args.output) is str else args.output[0]
file1 = args.input1 if type(args.input1) is str else args.input1[0]
file2 = args.input2 if type(args.input2) is str else args.input2[0]

numpoints = 500
dual_residual_use_cheb = np.zeros(shape=numpoints, dtype=np.float32)
dual_residual_no_cheb = np.zeros(shape=numpoints, dtype=np.float32)
with open(file1) as f1:
    ln = 0
    for line in f1.readlines():
        if '=' not in line:
            dual_residual_use_cheb[ln]  = line 
            ln += 1

with open(file2) as f2:
    ln = 0
    for line in f2.readlines():
        if '=' not in line:
            dual_residual_no_cheb[ln]  = line 
            ln += 1

assert(len(dual_residual_use_cheb) == len(dual_residual_no_cheb))

fig, ax = plt.subplots() 
fig.set_size_inches(20, 13)
x = np.linspace(0,numpoints,numpoints)
ax.plot(x, dual_residual_use_cheb, label='local_chebyshev')  # Plot some data on the axes.
ax.plot(x, dual_residual_no_cheb,  label='local_no_chebyshev')  # Plot more data on the axes...
ax.set_yscale("log")
ax.set_xlabel('timesteps')  # Add an x-label to the axes.
ax.set_ylabel('dual residual')  # Add a y-label to the axes.
ax.set_title("Comparison between chebyshev solver and without it")  # Add a title to the axes.
ax.legend()  # Add a legend.

plt.savefig(output_file)
plt.show()
