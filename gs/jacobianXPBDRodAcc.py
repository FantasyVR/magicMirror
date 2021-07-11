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
import numpy as np
from taichi.lang.ops import abs, sqrt

ti.init(arch=ti.gpu)
h = 0.01  # timestep size
squareInverseTime = 1.0/(h * h)
compliance = 1.0e-6
alpha = compliance * squareInverseTime
N = 10 # number of particles
NC = N - 1  # number of distance constraint
NStep = 2
NMaxIte = 20
omega = 1.0 # Relaxtion ratio
pos = ti.Vector.field(2, float, N)
oldPos = ti.Vector.field(2, float, N)
predictionPos = ti.Vector.field(2, float, N)
vel = ti.Vector.field(2, float, N)
disConsIdx = ti.Vector.field(2, int, NC)
disConsLen = ti.field(float, NC)
invmass = ti.field(dtype=float, shape=N)
lagrangian = ti.field(dtype=float, shape=NC)
gravity = ti.Vector([0, -0.98])
gradient = ti.Vector.field(2,float, 2 * NC)
deltaLagrangian = ti.field(float, NC)
# For validation
dualResidual = ti.field(float,())
primalResidual = ti.field(float,())

@ti.kernel
def initRod():
    for i in range(N):
        pos[i] = ti.Vector([0.5 + 0.1 * i, 0.7])
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
def resetLagrangin():
    # # reset lambda
    for i in range(NC):
        lagrangian[i] = 0.0

@ti.kernel
def solveConstraints():
    # solve constriants
    for i in range(NC):
        idx1,idx2 = disConsIdx[i]
        rest_len = disConsLen[i]
        invMass1 = invmass[idx1]
        invMass2 = invmass[idx2]
        sumInvMass = invMass1 + invMass2
        if sumInvMass < 1.0e-6:
            print("Wrong Mass Setting")
        p1, p2 = pos[idx1], pos[idx2]
        constraint = (p1 - p2).norm() - rest_len
        gradient[2 * i + 0] = (p1 - p2).normalized()
        gradient[2 * i + 1] = - gradient[2*i+0]
        deltaLagrangian[i] = -(constraint + lagrangian[i] * alpha) / (
            sumInvMass + alpha)
        lagrangian[i] = lagrangian[i] + deltaLagrangian[i]

@ti.kernel
def updatePos():
    # update positions
    for i in range(NC):
        idx1,idx2 = disConsIdx[i]
        invMass1 = invmass[idx1]
        invMass2 = invmass[idx2]
        if invMass1 != 0.0:
            pos[idx1] +=  omega * invMass1 * deltaLagrangian[i] * gradient[2 * i + 0]
        if invMass2 != 0.0:
            pos[idx2] +=  omega * invMass2 * deltaLagrangian[i] * gradient[2 * i + 1]

@ti.kernel
def updateV():
    # update velocities
    for i in range(N):
        if invmass[i] != 0.0:
            vel[i] = (pos[i] - oldPos[i]) / h

@ti.kernel
def computeResidual():
    dualResidual[None]  = 0.0
    primalResidual[None] = 0.0
    for i in range(NC):
        idx1,idx2 = disConsIdx[i]
        rest_len = disConsLen[i]
        invMass1 = invmass[idx1]
        invMass2 = invmass[idx2]
        p1, p2 = pos[idx1], pos[idx2]
        constraint = (p1 - p2).norm() - rest_len
        
        dualResidual[None] += abs(constraint - alpha * lagrangian[i])
        
        gradient = (p1 - p2).normalized()
        r0 = ti.Vector([0.0,0.0])
        r1 = r0
        if invMass1 != 0.0:
            r0 =   1.0 / invMass1 * (p1 -  predictionPos[idx1]) + lagrangian[i] * gradient
        if invMass2 != 0.0:
            r1 =  1.0 / invMass2 * (p2 -  predictionPos[idx2]) - lagrangian[i] * gradient
        primalResidual[None] += sqrt(r0.norm_sqr() + r1.norm_sqr())

    print("Dual Residual: ", dualResidual[None])
    print("Primal Residual: ", primalResidual[None])


initRod()
initConstraint()

gui = ti.GUI('XPBD')
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
            resetLagrangin()
            for ite in range(NMaxIte):
                solveConstraints()
                updatePos()
            updateV()
            computeResidual()

    position = pos.to_numpy()
    begin = position[:-1]
    end = position[1:]
    gui.lines(begin, end, radius=3, color=0x0000FF)
    gui.circles(pos.to_numpy(), radius=5, color=0xffaa33)
    gui.show()