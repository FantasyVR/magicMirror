import random 
import os 

N = 50
NV = (N+1)**2 * 2
file = "random.txt"
if os.path.exists(file):
    os.remove(file)
f = open(file, 'a+')
for i in range(NV):
    v = (random.random() - 0.5) * 0.1   
    f.write(f"{v}\n")
f.close()