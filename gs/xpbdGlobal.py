"""
xPBD Rod Simulation with Global System Matrix
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

pos = ti.Vector.field(2, float, N)
oldPos = ti.Vector.field(2, float, N)
predictionPos = ti.Vector.field(2, float, N)
vel = ti.Vector.field(2, float, N)
invmass = ti.field(float, N)

disConsIdx = ti.Vector.field(2, int, NC)
disConsLen = ti.field(float, NC)
gradient = ti.Vector.field(2, float, 2 * NC)
constraint = ti.field(float, NC)

#xpbd
compliance = 1.0e-6
alpha = compliance / h /h
lagrangian = ti.field(float, NC)

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
            vel[i] += h * gravity
            oldPos[i] = pos[i]
            pos[i] += h * vel[i]
            predictionPos[i] = pos[i]

@ti.kernel
def resetLambda():
    for i in range(NC):
        lagrangian[i] = 0.0

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
        #constraint[i] = l - rest_len
        # xpbd
        constraint[i] = l - rest_len + alpha * lagrangian[i]
        gradient[2 * i + 0] = (p1 - p2).normalized()
        gradient[2 * i + 1] = -gradient[2 * i + 0]


def assemble(mass, g, c, cidx):
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
        A[start + i, 2 * idx1:2 * idx1 + 2] = g0
        A[start + i, 2 * idx2:2 * idx2 + 2] = g1
        #  uppper right
        A[2 * idx1:2 * idx1 + 2, start + i] = -g0
        A[2 * idx2:2 * idx2 + 2, start + i] = -g1
        # xpbd lower right
        A[start + i, start + i] = alpha

    np.set_printoptions(precision=4, suppress=True)

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

    x = np.linalg.solve(A[2:, 2:], b[2:])
    return x


@ti.kernel
def updatePos(x: ti.ext_arr()):
    for i in range(N - 1):
        pos[i + 1] += ti.Vector([x[2 * i + 0], x[2 * i + 1]])
    
@ti.kernel
def updateLambda(x: ti.ext_arr()):
    for i in range(NC):
        lagrangian[i] = x[2*N + i]

@ti.kernel
def updateV():
    for i in range(N):
        if invmass[i] != 0.0:
            vel[i] = (pos[i] - oldPos[i]) / h


initRod()
initConstraint()
gui = ti.GUI('xPBD with Global System')
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
            resetLambda()
            for ite in range(NMaxIte):
                computeCg()
                x = assemble(invmass.to_numpy(), gradient.to_numpy(),
                             constraint.to_numpy(), disConsIdx.to_numpy())
                updatePos(x)
                updateLambda(x)
            updateV()
    position = pos.to_numpy()
    begin = position[:-1]
    end = position[1:]
    gui.lines(begin, end, radius=3, color=0x0000FF)
    gui.circles(pos.to_numpy(), radius=5, color=0xffaa33)
    gui.show()