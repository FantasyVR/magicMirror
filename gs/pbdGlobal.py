"""
Add geometric stiffness into XPBD

The process divives into two parts:
Part I: taichi side
1. compute system matrix submodule such as mass matrix(using vector or scalar), gradient matrix (using vectors of vector), compliance matrix
2. compute right hand side vectors 
Part II: scipy and numpy side
1. copy submodule matrix to numpy array and assmable a global system sparse matrix with scipy 
2. use scipy to solve the `Ax = b`
3. update field with external array
"""
import taichi as ti
from taichi.lang.ops import abs, sqrt
import numpy as np

ti.init(arch=ti.cpu)
# general setting
gravity = ti.Vector([0, -9.8])
h = 0.1  # timestep size
N = 3  # number of particles
NC = N - 1  # number of distance constraint
NStep = 2
NMaxIte = 3

pos             = ti.Vector.field(2, float, N)
oldPos          = ti.Vector.field(2, float, N)
predictionPos   = ti.Vector.field(2, float, N)
vel             = ti.Vector.field(2, float, N)
invmass         = ti.field(float, N)

disConsIdx  = ti.Vector.field(2, int, NC)
disConsLen  = ti.field(float, NC)
gradient    = ti.Vector.field(2, float, 2 * NC)
constraint  = ti.field(float, NC)


@ti.kernel
def initRod():
    for i in range(N):
        pos[i] = ti.Vector([0.5 + 0.1 * i, 0.7])
        # pos[i] = ti.Vector([0.5, 0.5- 0.1 * i])
        oldPos[i] = pos[i]
        vel[i] = ti.Vector([0.0, 0.0])
        invmass[i] = 1.0
    invmass[0] = 0.0  # set the first particle static


@ti.kernel
def initConstraint():
    for i in range(NC):
        disConsIdx[i] = ti.Vector([i, i + 1])
        disConsLen[i] = (pos[i + 1] - pos[i]).norm()

@ti.kernel
def semiEuler():
    # semi-euler update pos & vel
    for i in range(N):
        if (invmass[i] != 0.0):
            vel[i] +=  h * gravity
            oldPos[i] = pos[i]
            pos[i] += h * vel[i]
            predictionPos[i] = pos[i]


# compute constraint vector and gradient vector 
@ti.kernel
def computeCg():
    for i in range(NC):
        idx1, idx2 = disConsIdx[i]
        rest_len = disConsLen[i]
        invMass1 = invmass[idx1]
        invMass2 = invmass[idx2]
        sumInvMass = invMass1 + invMass2
        if sumInvMass < 1.0e-6:
            print("Wrong Mass Setting")
        p1, p2 = pos[idx1], pos[idx2]
        l = (p1 - p2).norm()
        constraint[i] = l - rest_len 
        gradient[2 * i + 0] = (p1 - p2).normalized()
        gradient[2 * i + 1] = -gradient[2 * i + 0]

def assemble(mass, g,  c, cidx):
    dim = (2 * N + NC)  # the system dimension

    A = np.zeros((dim, dim), dtype=np.float32)
    # uppper left: mass matrix + geometric stiffness matrix
    for i in range(N):
        A[2 * i, 2 * i] = mass[i]
        A[2 * i + 1, 2 * i + 1] = mass[i]

    # Other parts
    start = 2 * N
    for i in range(NC):
        idx1, idx2 = cidx[i]
        g0 = g[2 * i + 0]
        g1 = g[2 * i + 1]
        #  lower left
        A[start + i, 2 * idx1:2 * idx1 + 2] =g0
        A[start + i, 2 * idx2:2 * idx2 + 2] =g1
        #  uppper right
        A[2 * idx1:2 * idx1 + 2, start + i] =- g0
        A[2 * idx2:2 * idx2 + 2, start + i] =- g1
    
    np.set_printoptions(precision=4,suppress = True)

    G = np.zeros((2 * N, NC))
    for i in range(NC):
        idx1, idx2 = cidx[i]
        g0 = g[2 * i + 0]
        g1 = g[2 * i + 1]
        G[2 * idx1:2 * idx1 + 2, i] += g0
        G[2 * idx2:2 * idx2 + 2, i] += g1
    # RHS
    b = np.zeros(dim, dtype=np.float32)
    b[2 * N:] = -c 

    print(f"Real system matrix: \n {A} \n\n-----------------------")
    print(f"Real right hand side: \n{b} \n\n-------------------------")
    print("rank(G) = ", np.linalg.matrix_rank(G),"; real rank should be: ", NC)
    print("rank(A) = ", np.linalg.matrix_rank(A))
    x = np.linalg.solve(A[2:, 2:], b[2:])
    print(f"Solusion: {x}")
    return x
 

@ti.kernel
def updatePos(x: ti.ext_arr()):
    for i in range(N-1):
        print("Before add pos[i+1]: ", pos[i+1])
        print("x: [",x[2*i+0], " ",x[2*i+1],"]" )
        pos[i+1] += ti.Vector([x[2 * i + 0], x[2 * i + 1]])
        print("After add pos[i+1]: ", pos[i+1])

@ti.kernel
def updateV():
    for i in range(N):
        if invmass[i] != 0.0:
            vel[i] = (pos[i] - oldPos[i]) / h
            print("pos[",i,"]: ", pos[i])
            print("oldPos[",i,"]: ", oldPos[i])
            print("V[", i, "]: ", vel[i])


initRod()
initConstraint()
# np.printoptions(precision=4,suppress=True)
# print(f"before simualtion, the position is : {pos.to_numpy()}")
# for i in range(NStep):
#     semiEuler()
#     print(f"================ step {i} ===============\n: {pos.to_numpy()} \n==================")
#     for ite in range(NMaxIte):
#         computeCg()
#         x = assemble(invmass.to_numpy(), gradient.to_numpy(),constraint.to_numpy(),disConsIdx.to_numpy())
#         updatePos(x)

#     updateV()
#     print("\n\n\n")
gui = ti.GUI('XPBD with Global System')
pause = True
while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        if e.key == gui.SPACE:
            pause = not pause
    if not pause:
        for i in range(NStep):
            semiEuler()
            for ite in range(NMaxIte):
                computeCg()
                x = assemble(invmass.to_numpy(), gradient.to_numpy(),
                            constraint.to_numpy(),
                            disConsIdx.to_numpy())
                updatePos(x)
            updateV()
    position = pos.to_numpy()
    begin = position[:-1]
    end = position[1:]
    gui.lines(begin, end, radius=3, color=0x0000FF)
    gui.circles(pos.to_numpy(), radius=5, color=0xffaa33)
    gui.show()
