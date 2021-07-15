"""
Stable Constrainted Dynamics, Maxime Tournier et.al, 2015.
"""
import taichi as ti
from taichi.lang.ops import abs, sqrt
import numpy as np
import os
ti.init(arch=ti.cpu)
# general setting
gravity = ti.Vector([0, -9.8])
h = 0.01  # timestep size
N = 3  # number of particles
NC = N - 1  # number of distance constraint
NStep = 5
NMaxIte = 2

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
alpha = compliance / h / h
lagrangian = ti.field(float, NC)

# geometric stiffness
K = ti.Matrix.field(2, 2, float, (N, N))

# For validation
dualResidual = ti.field(float, ())
primalResidual = ti.field(float, ())
maxdualResidual = ti.field(float, ())
maxprimalResidual = ti.field(float, ())


@ti.kernel
def initRod():
    for i in range(N):
        # pos[i] = ti.Vector([0.5 + 0.1 * i, 0.7])
        pos[i] = ti.Vector([0.1 * i, 0.0])
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
            vel[i] = vel[i] + h * gravity
            oldPos[i] = pos[i]
            pos[i] = pos[i] + h * vel[i]
            predictionPos[i] = pos[i]


@ti.kernel
def resetLambda():
    for i in range(NC):
        lagrangian[i] = 0.0


@ti.kernel
def resetK():
    for i, j in ti.ndrange(N, N):
        K[i, j] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])


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
        n = (p1 - p2).normalized()
        # xpbd
        constraint[i] = l - rest_len + alpha * lagrangian[i]
        gradient[2 * i + 0] = n
        gradient[2 * i + 1] = -n
        # geometric stiffness
        print("p[", idx1,"]:", p1)
        print("p[", idx2,"]:", p2)
        print("Lambda: ", lagrangian[i])
        print("gradient: ", n)
        I = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
        k = -lagrangian[i] / l * (I - n @ n.transpose())
        print("k: ", k)
        K[idx1, idx1] -= k
        K[idx1, idx2] += k
        K[idx2, idx1] += k
        K[idx2, idx2] -= k


def assemble(mass, p, prep, g, KK, l, c, cidx):
    dim = (2 * N + NC)  # the system dimension

    A = np.zeros((dim, dim), dtype=np.float32)
    # uppper left: mass matrix
    for i in range(N):
        A[2 * i, 2 * i] = mass[i]
        A[2 * i + 1, 2 * i + 1] = mass[i]

    # uppper left: geometric stiffness
    for i in range(N):
        for j in range(N):
            A[2 * i:2 * i + 2, 2 * j:2 * j + 2] += KK[i, j]
    # print("K matrix: \n", KK)
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

    # geometric stiffness
    G = np.zeros((2 * N, NC))
    for i in range(NC):
        idx1, idx2 = cidx[i]
        g0 = g[2 * i + 0]
        g1 = g[2 * i + 1]
        G[2 * idx1:2 * idx1 + 2, i] += g0
        G[2 * idx2:2 * idx2 + 2, i] += g1
    print("G: \n", G.T)
    print("l: ", l)
    Gl = G @ l

    # RHS
    b = np.zeros(dim, dtype=np.float32)
    b[2 * N:] = -c
    # geometric stiffness
    for i in range(N):
        b[2 * i:2 * i + 2] = -mass[i] * (p[i] - prep[i])
    b[:2 * N] += Gl
    print("A: \n", A)
    print("b: ", b)
    x = np.linalg.solve(A[2:, 2:], b[2:])
    print("Solusion: ", x)
    return x


@ti.kernel
def updatePos(x: ti.ext_arr()):
    for i in range(N - 1):
        pos[i + 1] += ti.Vector([x[2 * i + 0], x[2 * i + 1]])
    print("Update Position: ")
    for i in range(N):
        print(pos[i])


@ti.kernel
def updateLambda(x: ti.ext_arr()):
    for i in range(NC):
        lagrangian[i] += x[2 * (N - 1) + i]


@ti.kernel
def updateV():
    for i in range(N):
        if invmass[i] != 0.0:
            vel[i] = (pos[i] - oldPos[i]) / h


@ti.kernel
def computeResidual():
    dualResidual[None] = 0.0
    primalResidual[None] = 0.0
    for i in range(NC):
        idx1, idx2 = disConsIdx[i]
        rest_len = disConsLen[i]
        invMass1 = invmass[idx1]
        invMass2 = invmass[idx2]
        p1, p2 = pos[idx1], pos[idx2]
        constraint = (p1 - p2).norm() - rest_len

        dualResidual[None] += abs(constraint - alpha * lagrangian[i])

        gradient = (p1 - p2).normalized()
        r0 = ti.Vector([0.0, 0.0])
        r1 = r0
        if invMass1 != 0.0:
            r0 = 1.0 / invMass1 * (
                p1 - predictionPos[idx1]) + lagrangian[i] * gradient
        if invMass2 != 0.0:
            r1 = 1.0 / invMass2 * (
                p2 - predictionPos[idx2]) - lagrangian[i] * gradient
        primalResidual[None] += sqrt(r0.norm_sqr() + r1.norm_sqr())
    if maxdualResidual[None] < dualResidual[None]:
        maxdualResidual[None] = dualResidual[None]
    if maxprimalResidual[None] < primalResidual[None]:
        maxprimalResidual[None] = primalResidual[None]

    print("Dual Residual: ", dualResidual[None])
    print("Primal Residual: ", primalResidual[None])


initRod()
initConstraint()
for i in range(NStep):
    print(f"\n-----------------------------start time step {i}-------------------------------")
    semiEuler()
    resetLambda()
    resetK()
    for ite in range(NMaxIte):
        print(f"\n-----------------------------start iteration  {ite}-------------------------------")
        computeCg()
        print("========================Start assmble matrix===================")
        x = assemble(invmass.to_numpy(), pos.to_numpy(),
                     predictionPos.to_numpy(),
                     gradient.to_numpy(), K.to_numpy(), lagrangian.to_numpy(),
                     constraint.to_numpy(), disConsIdx.to_numpy())
        print("=========================Ending assmble matrix=================\n")
        updatePos(x)
        updateLambda(x)
        print("Lagrangian Multipler: ", lagrangian.to_numpy())
        print(f"\n-----------------------------end iteration  {ite}-------------------------------")
    updateV()
    print("After solusiton V:", vel.to_numpy())

# gui = ti.GUI('Stable Constrainted Dynamics')
# pause = True
# while gui.running:
#     for e in gui.get_events(gui.PRESS):
#         if e.key == gui.ESCAPE:
#             gui.running = False
#         if e.key == gui.SPACE:
#             pause = not pause
#     if not pause:
#         for i in range(NStep):
#             semiEuler()
#             resetLambda()
#             resetK()
#             for ite in range(NMaxIte):
#                 computeCg()
#                 x = assemble(invmass.to_numpy(), pos.to_numpy(),
#                              predictionPos.to_numpy(), gradient.to_numpy(),
#                              K.to_numpy(), lagrangian.to_numpy(),
#                              constraint.to_numpy(), disConsIdx.to_numpy())
#                 updatePos(x)
#                 updateLambda(x)
#             updateV()
#             computeResidual()
#         print("max dual residual: ", maxdualResidual.to_numpy())
#         print("max prim residual: ", maxprimalResidual.to_numpy())

#     position = pos.to_numpy()
#     begin = position[:-1]
#     end = position[1:]
#     gui.lines(begin, end, radius=3, color=0x0000FF)
#     gui.circles(pos.to_numpy(), radius=5, color=0xffaa33)
#     gui.show()
