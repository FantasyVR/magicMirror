"""
We use XPBD(gpu version) to simulate springs
"""
import taichi as ti
from taichi.lang.ops import abs, sqrt

ti.init(arch=ti.cpu)
gravity = ti.Vector([0, -0.98])
h = 0.01  # timestep size

NStep = 4
NMaxIte = 100
N = 5 # number of particles
NC = N - 1  # number of distance constraint

omega = 1.0 # Relaxtion ratio
pos = ti.Vector.field(2, float, N)
accpos = ti.Vector.field(2, float, N)
oldPos = ti.Vector.field(2, float, N)
predictionPos = ti.Vector.field(2, float, N)
vel = ti.Vector.field(2, float, N)
disConsIdx = ti.Vector.field(2, int, NC)
disConsLen = ti.field(float, NC)
invmass = ti.field(dtype=float, shape=N)
lagrangian = ti.field(dtype=float, shape=NC)

# xpbd
squareInverseTime = 1.0/(h * h)
compliance = 1.0e-6
alpha = compliance * squareInverseTime
valpha = ti.field(float,NC)
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
    invmass[N-1] = 0.01  # set the first particle static


@ti.kernel
def initConstraint():
    for i in range(NC):
        disConsIdx[i] = ti.Vector([i, i + 1])
        disConsLen[i] = (pos[i + 1] - pos[i]).norm()
        valpha[i] = alpha
    valpha[NC-1] = 100 * alpha

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
def resetAccPos():
    for i in range(N):
        accpos[i] = ti.Vector([0.0,0.0])

@ti.kernel
def update(ts:ti.i32, ite:ti.i32):
    # solve constriant
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
        gradient = (p1 - p2).normalized()
        deltaLagrangian = -(constraint + lagrangian[i] * valpha[i]) / (
            sumInvMass + valpha[i])
        lagrangian[i] += deltaLagrangian
        # print("[",ts, ",", ite,"], dL", i , ":", deltaLagrangian,", lambda:", lagrangian[i])
        if invMass1 != 0.0:
            accpos[idx1] +=  omega * invMass1 * deltaLagrangian * gradient
        if invMass2 != 0.0:
            accpos[idx2] += -omega * invMass2 * deltaLagrangian * gradient

@ti.kernel 
def updateP(): 
    #update positions and velocities
    for i in range(N):
        if (invmass[i] != 0.0):
            pos[i] += accpos[i]
            # print(i, "-th pos: ", pos[i])

@ti.kernel
def updateV():
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
        
        dualResidual[None] += abs(constraint - valpha[i] * lagrangian[i])
        
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
# for i in range(NStep):
#     semiEuler()
#     resetLagrangin()
#     #resetAccPos()
#     for ite in range(NMaxIte):
#         resetAccPos()
#         update(i,ite)
#         updateP()
#     updateV()
#     computeResidual()


gui = ti.GUI('XPBD')
pause = False
frame = 0
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
                resetAccPos()
                update(i,ite)
                updateP()
            updateV()
            computeResidual()

    position = pos.to_numpy()
    begin = position[:-1]
    end = position[1:]
    gui.lines(begin, end, radius=3, color=0x0000FF)
    gui.circles(pos.to_numpy(), radius=5, color=0xffaa33)
    filename = f'./data/frame_{frame:05d}.png'   # create filename with suffix png
    frame += 1
    if frame == 500:
        break
    gui.show(filename)