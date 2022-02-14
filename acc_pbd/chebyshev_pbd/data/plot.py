import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Acc stable constrainted dyanmics')
parser.add_argument('--output', '-o', nargs=1, default="data/comparision.png", type=str, required=False)
parser.add_argument('--input1', '-i1',nargs=1, default="data/chebyshev_dual_residual.txt", type=str, required=False)
parser.add_argument('--input2', '-i2',nargs=1, default="data/dual_residual.txt", type=str, required=False)
parser.add_argument('--lable1', '-l1',nargs=1, default="with_primal_chebyshev", type=str, required=False)
parser.add_argument('--lable2', '-l2',nargs=1, default="no_primal_chebyshev", type=str, required=False)


args = parser.parse_args()
output_file = args.output if type(args.output) is str else args.output[0]
file1 = args.input1 if type(args.input1) is str else args.input1[0]
file2 = args.input2 if type(args.input2) is str else args.input2[0]
lable1 = args.lable1 if type(args.lable1) is str else args.lable1[0]
lable2 = args.lable2 if type(args.lable2) is str else args.lable2[0]


numpoints = 1000
start_frame = 0
dual_residual_use_cheb = np.zeros(shape=numpoints, dtype=np.float32)
dual_residual_no_cheb = np.zeros(shape=numpoints, dtype=np.float32)
with open(file1) as f1:
    ln = 0
    for line in f1.readlines():
        if ln >= start_frame and ln < start_frame + numpoints:
            dual_residual_use_cheb[ln-start_frame]  = line 
        ln += 1

with open(file2) as f2:
    ln = 0
    for line in f2.readlines():
        if ln >= start_frame and ln < start_frame + numpoints:
            dual_residual_no_cheb[ln-start_frame]  = line 
        ln += 1

assert(len(dual_residual_use_cheb) == len(dual_residual_no_cheb))

fig, ax = plt.subplots() 
fig.set_size_inches(20, 13)
x = np.linspace(0,numpoints,numpoints)
ax.plot(x, dual_residual_use_cheb, label=lable1)  # Plot some data on the axes.
ax.plot(x, dual_residual_no_cheb,  label=lable2)  # Plot more data on the axes...
ax.set_yscale("log")
ax.set_xlabel('timesteps')  # Add an x-label to the axes.
ax.set_ylabel('dual residual')  # Add a y-label to the axes.
ax.set_title("PBD vs PBD+Chebyshev")  # Add a title to the axes.
ax.legend()  # Add a legend.

plt.savefig(output_file)
