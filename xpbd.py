"""
We use XPBD(cpu version) to simulate springs
"""
import taichi as ti

ti.init(arch=ti.cpu)
h = 0.01  # timestep size
compliance = 1.0e-6
alpha = compliance * (1.0 / h / h)
N = 4  # number of particles
NC = N - 1  # number of distance constraint
NStep = 5
NMaxIte = 50
pos = ti.Vector.field(2, float, N)
oldPos = ti.Vector.field(2, float, N)
vel = ti.Vector.field(2, float, N)
disConsIdx = ti.Vector.field(2, int, NC)
disConsLen = ti.field(float, NC)
invmass = ti.field(dtype=float, shape=N)
lagrangian = ti.field(dtype=float, shape=NC)
gravity = ti.Vector([0, -0.98])


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


@ti.func
def update():
    # semi-euler update pos & vel
    for i in range(N):
        if (invmass[i] != 0.0):
            vel[i] += h * gravity
            oldPos[i] = pos[i]
            pos[i] += h * vel[i]
    # # reset lambda
    for i in range(NC):
        lagrangian[i] = 0.0
    # # solve constriant
    for ite in range(NMaxIte):
        for i in range(NC):
            x,y = disConsIdx[i]
            index = disConsIdx[i]
            rest_len = disConsLen[i]
            idx1 = index[0]
            idx2 = index[1]
            invMass1 = invmass[idx1]
            invMass2 = invmass[idx2]
            sumInvMass = invMass1 + invMass2
            if sumInvMass < 1.0e-6:
                print("Wrong Mass Setting")
            p1 = pos[idx1]
            p2 = pos[idx2]
            constraint = (p1 - p2).norm() - rest_len
            gradient = (p1 - p2).normalized()
            deltaLagrangian = -(constraint + lagrangian[i] * alpha) / (
                sumInvMass + alpha)
            lagrangian[i] += deltaLagrangian
            if invMass1 != 0.0:
                pos[idx1] += invMass1 * deltaLagrangian * gradient
            if invMass2 != 0.0:
                pos[idx2] += -invMass2 * deltaLagrangian * gradient
    # # update pos and vel
    for i in range(N):
        if (invmass[i] != 0.0):
            vel[i] = (pos[i] - oldPos[i]) / h

@ti.kernel
def timestep():
    for i in range(NStep):
        update()

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
        timestep()

    position = pos.to_numpy()
    begin = position[:-1]
    end = position[1:]
    gui.lines(begin, end, radius=3, color=0x0000FF)
    gui.circles(pos.to_numpy(), radius=5, color=0xffaa33)
    gui.show()