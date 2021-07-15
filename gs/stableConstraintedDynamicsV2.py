"""
Rod simulation based on [Stable Constrainted Dynamics, Maxime Tournier et.al, 2015.]
"""
import taichi as ti
from taichi.lang.ops import abs, sqrt
import numpy as np

ti.init(arch=ti.cpu)
gravity = ti.Vector([0, -9.8])
h = 0.01  # timestep size

NStep = 5  # number of steps in each frame
NMaxIte = 1  # number of iterations in each step
N = 10  # number of particles
NC = N - 1  # number of distance constraint

pos = ti.Vector.field(2, ti.f64, N)
oldPos = ti.Vector.field(2, ti.f64, N)
predictionPos = ti.Vector.field(2, ti.f64, N)
vel = ti.Vector.field(2, ti.f64, N)
mass = ti.field(ti.f64, N)

disConsIdx = ti.Vector.field(
    2, int, NC)  # each element store vertex indices of the constraint
disConsLen = ti.field(
    ti.f64, NC
)  # rest state (rest length of spring in this example) of each constraint
gradient = ti.Vector.field(2, ti.f64, 2 * NC)  # gradient of constraints
constraint = ti.field(ti.f64, NC)  # constraints violation

#xpbd values
compliance = 1.0e-6
alpha = compliance / h / h
lagrangian = ti.field(ti.f64, NC)

# geometric stiffness
K = ti.Matrix.field(2, 2, ti.f64, (N, N))

# For validation
dualResidual = ti.field(ti.f64, ())
primalResidual = ti.field(ti.f64, ())
maxdualResidual = ti.field(ti.f64, ())
maxprimalResidual = ti.field(ti.f64, ())

# For line search
tpos = ti.Vector.field(2, ti.f64, N)
tlag = ti.field(ti.f64, NC)


@ti.kernel
def initRod():
    for i in range(N):
        pos[i] = ti.Vector([0.5 + 0.1 * i, 0.7])
        # pos[i] = ti.Vector([0.1 * i, 0.0])
        # pos[i] = ti.Vector([0.5, 0.5- 0.1 * i])
        oldPos[i] = pos[i]
        vel[i] = ti.Vector([0.0, 0.0])
        mass[i] = 1.0
    mass[0] = 0.0  # set the first particle static
    mass[9] = 100.0


@ti.kernel
def initConstraint():
    for i in range(NC):
        disConsIdx[i] = ti.Vector([i, i + 1])
        disConsLen[i] = (pos[i + 1] - pos[i]).norm()


@ti.kernel
def semiEuler():
    # semi-euler update pos & vel
    for i in range(N):
        if (mass[i] != 0.0):
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
        invMass1 = mass[idx1]
        invMass2 = mass[idx2]
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
        """
            k = lambda[i]/l * (I - n * n')
            K = | Hessian_{x1,x1}, Hessian_{x1,x2}   |  = | k  -k|
                | Hessian_{x1,x2}, Hessian_{x2,x2}   |    |-k   k|
        """
        I = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
        k = lagrangian[i] / l * (I - n @ n.transpose())
        K[idx1, idx1] += k
        K[idx1, idx2] -= k
        K[idx2, idx1] -= k
        K[idx2, idx2] += k


"""
Assemble system matrix
A =   |  M - K     -J' |
      |  J       alpha |
b = |      -M(x - y) + J' * lambda      |
    | -(constraint + alpha * lambda)    |
"""


def assemble(mass, p, prep, g, KK, l, c, cidx, step):
    dim = (2 * N + NC)  # the system dimension

    A = np.zeros((dim, dim), dtype=np.float64)
    # uppper left: mass matrix
    for i in range(N):
        A[2 * i, 2 * i] = mass[i]
        A[2 * i + 1, 2 * i + 1] = mass[i]

    # uppper left: geometric stiffness
    if step != 0:
        for i in range(N):
            for j in range(N):
                A[2 * i:2 * i + 2, 2 * j:2 * j + 2] -= KK[i, j]
    # print("K matrix: \n", repr(KK))
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

    # geometric stiffness :
    G = np.zeros((2 * N, NC))
    for i in range(NC):
        idx1, idx2 = cidx[i]
        g0 = g[2 * i + 0]
        g1 = g[2 * i + 1]
        G[2 * idx1:2 * idx1 + 2, i] = g0
        G[2 * idx2:2 * idx2 + 2, i] = g1
    Gl = G @ l
    # RHS
    b = np.zeros(dim, dtype=np.float64)
    b[2 * N:] = -c
    # geometric stiffness
    for i in range(1, N):
        b[2 * i:2 * i + 2] = -mass[i] * (p[i] - prep[i])
    b[:2 * N] += Gl
    # np.set_printoptions(precision=5, suppress=False)
    # print("b: ", b)
    x = np.linalg.solve(A[2:, 2:], b[2:])
    return x


@ti.kernel
def updateV():
    for i in range(N):
        if mass[i] != 0.0:
            vel[i] = (pos[i] - oldPos[i]) / h


@ti.kernel
def computeResidual(step: ti.i32):
    dualResidual[None] = 0.0
    primalResidual[None] = 0.0
    for i in range(NC):
        idx1, idx2 = disConsIdx[i]
        rest_len = disConsLen[i]
        invMass1 = mass[idx1]
        invMass2 = mass[idx2]
        p1, p2 = pos[idx1], pos[idx2]
        constraint = (p1 - p2).norm() - rest_len

        dualResidual[None] += abs(constraint + alpha * lagrangian[i])

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
    print("-----step:", step, "-------------")
    print("Dual Residual: ", dualResidual[None])
    print("Primal Residual: ", primalResidual[None])


@ti.kernel
def computeEnergy() -> ti.f64:
    e = 0.0
    for i in range(N):
        dx = tpos[i] - predictionPos[i]
        e += mass[i] * dx.norm_sqr()
    for i in range(NC):
        idx1, idx2 = disConsIdx[i]
        rest_len = disConsLen[i]
        p1, p2 = tpos[idx1], tpos[idx2]
        l = (p1 - p2).norm()
        # xpbd
        c = l - rest_len + alpha * tlag[i]
        e += alpha * c * c
    return e


@ti.kernel
def setTposTlag(stepsize: ti.f64, dx: ti.ext_arr()):
    tpos[0] = pos[0]  # static position
    for i in range(N - 1):
        tpos[i +
             1] = pos[i +
                      1] + stepsize * ti.Vector([dx[2 * i + 0], dx[2 * i + 1]])
    for i in range(NC):
        tlag[i] = lagrangian[i] + stepsize * dx[2 * (N - 1) + i]


@ti.kernel
def updatePosLambda():
    for i in range(N):
        pos[i] = tpos[i]
    for i in range(NC):
        lagrangian[i] = tlag[i]


initRod()
initConstraint()
gui = ti.GUI('Stable Constrainted Dynamics')
pause = True
while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        if e.key == gui.SPACE:
            pause = not pause
    if not pause:
        # resetLambda()
        for step in range(NStep):
            semiEuler()
            # resetLambda()
            for ite in range(NMaxIte):
                resetK()
                computeCg()
                dx = assemble(mass.to_numpy(), pos.to_numpy(),
                              predictionPos.to_numpy(), gradient.to_numpy(),
                              K.to_numpy(), lagrangian.to_numpy(),
                              constraint.to_numpy(), disConsIdx.to_numpy(),
                              step)

                # line search part
                # setTposTlag(0.0, dx)
                # Energy  = computeEnergy()
                # print(f"Energy: {Energy}")
                # lsSize = 1.0 # line search step size
                # while True:
                #     print(f"step size: {lsSize}")
                #     setTposTlag(lsSize, dx)
                #     E = computeEnergy()
                #     print(f"e: {E}")
                #     if E <= Energy or E < 1.0e-6:
                #         break
                #     lsSize = 0.5 * lsSize

                setTposTlag(1.0, dx)
                updatePosLambda()
            computeResidual(step)
            updateV()
        # print("max dual residual: ", maxdualResidual.to_numpy())
        # print("max prim residual: ", maxprimalResidual.to_numpy())

    position = pos.to_numpy()
    begin = position[:-1]
    end = position[1:]
    gui.lines(begin, end, radius=3, color=0x0000FF)
    gui.circles(pos.to_numpy(), radius=5, color=0xffaa33)
    gui.show()

# valadition without GUI
# for i in range(NStep):
#     print(f"########################################start time step {i+1}##########################################")
#     semiEuler()
#     resetLambda()
#     for ite in range(NMaxIte):
#         print(f"-----------------------------start iteration  {ite+1}-------------------------------")
#         print("position: ", repr(np.reshape(pos.to_numpy(),(1,2*N))))
#         if i == 20 and ite == 10:
#             np.savetxt('data/pos.txt', np.reshape(pos.to_numpy(),(1,2*N)), delimiter=',')
#             np.savetxt('data/lambda.txt',np.reshape(lagrangian.to_numpy(),(1,NC)),delimiter=',')

#         resetK()
#         computeCg()
#         # print("========================Start assmble matrix===================")
#         x = assemble(mass.to_numpy(), pos.to_numpy(),
#                      predictionPos.to_numpy(),
#                      gradient.to_numpy(), K.to_numpy(), lagrangian.to_numpy(),
#                      constraint.to_numpy(), disConsIdx.to_numpy())
#         # print("=========================Ending assmble matrix=================\n")
#         updatePos(x)
#         print("Corrected position:", repr(np.reshape(pos.to_numpy(),(1,2*N))))
#         updateLambda(x)
#         print("Sum lambda: ", repr(np.reshape(lagrangian.to_numpy(),(1,NC))))

#         if i == 20 and ite == 10:
#             np.savetxt('data/updatedPos.txt', np.reshape(pos.to_numpy(),(1,2*N)), delimiter=',')
#             np.savetxt('data/updateLambda.txt',np.reshape(lagrangian.to_numpy(),(1,NC)),delimiter=',')

#         # print("Lagrangian Multipler: ", lagrangian.to_numpy())
#         # print(f"\n-----------------------------end iteration  {ite}-------------------------------")
#     print("Old position:", repr(np.reshape(oldPos.to_numpy(),(1,2*N))))
#     updateV()
#     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Velocities: ", repr(np.reshape(vel.to_numpy(),(1,2*N))))
