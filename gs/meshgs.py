"""
Mass-spring system simulation based on [Stable Constrainted Dynamics, Maxime Tournier et.al, 2015.]
"""
import taichi as ti
from taichi.lang.ops import abs, sqrt
import numpy as np

ti.init(arch=ti.cpu)
gravity = ti.Vector([0, -9.8])
h = 0.01  # timestep size

NStep = 5  # number of steps in each frame
NMaxIte = 1  # number of iterations in each step
N = 5  # number of particles
NE = (2 * N + 1) * N * 2 # number of edges
NV = (N + 1)**2  # number of vertices

pos = ti.Vector.field(2, ti.f64, NV)
oldPos = ti.Vector.field(2, ti.f64, NV)
predictionPos = ti.Vector.field(2, ti.f64, NV)
vel = ti.Vector.field(2, ti.f64, NV)
mass = ti.field(ti.f64, NV)

disConsIdx = ti.Vector.field(2, int, NE)  # each element store vertex indices of the constraint
disConsLen = ti.field(ti.f64, NE)  # rest state (rest length of spring in this example) of each constraint
gradient = ti.Vector.field(2, ti.f64, 2 * NE)  # gradient of constraints
constraint = ti.field(ti.f64, NE)  # constraints violation

#xpbd values
compliance = 1.0e-5
alpha = compliance / h / h
lagrangian = ti.field(ti.f64, NE)

# geometric stiffness
K = ti.Matrix.field(2, 2, ti.f64, (NV, NV))

# For validation
dualResidual = ti.field(ti.f64, ())
primalResidual = ti.field(ti.f64, ())
maxdualResidual = ti.field(ti.f64, ())
maxprimalResidual = ti.field(ti.f64, ())

# For line search
tpos = ti.Vector.field(2, ti.f64, NV)
tlag = ti.field(ti.f64, NE)

# For control
attractor_pos = ti.Vector.field(2, float, ())
attractor_strength = ti.field(float, ())

@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, N + 1):
        k = i * (N + 1) + j
        pos[k] = ti.Vector([i, j]) * 0.05 + ti.Vector([0.4, 0.4])
        oldPos[k] = pos[k]
        vel[k] = ti.Vector([0, 0])
        mass[k] = 1.0
    mass[0] = 0.0
    # for i in range(N + 1):
    #     k = i * (N + 1) + N
    #     mass[k] = 0.0
    # k0 = N
    # k1 = (N + 2) * N
    # mass[k0] = 0.0
    # mass[k1] = 0.0


@ti.kernel
def init_mesh():
    for i, j in ti.ndrange(N+1,N):
        # horizontal
        a = i * (N + 1) + j
        disConsIdx[i * N + j] = ti.Vector([a, a+1])
    start = N * (N + 1)
    for i,j in ti.ndrange(N,N+1):
        # vertical
        a = i * (N + 1) + j
        disConsIdx[start + i + j * N] = ti.Vector([a,a+N+1])
    start = 2 * start
    for i, j in ti.ndrange(N, N):
        # diagonal
        a = i * (N + 1) + j
        disConsIdx[start + i * N + j] = ti.Vector([a, a+N+2])
    start += N * N 
    for i, j in ti.ndrange(N, N):
        a = i * (N + 1) + j
        disConsIdx[start + i * N + j] = ti.Vector([a+1,a+N+1])

@ti.kernel
def initConstraint():
    for i in range(NE):
        a, b = disConsIdx[i] 
        disConsLen[i] = (pos[a] - pos[b]).norm()


@ti.kernel
def semiEuler():
    # semi-euler update pos & vel
    for i in range(NV):
        if (mass[i] != 0.0):
            vel[i] = vel[i] + h * gravity + attractor_strength[None] * (
                attractor_pos[None] - pos[i]).normalized(1e-5)
            oldPos[i] = pos[i]
            pos[i] = pos[i] + h * vel[i]
            predictionPos[i] = pos[i]


@ti.kernel
def resetLambda():
    for i in range(NE):
        lagrangian[i] = 0.0


@ti.kernel
def resetK():
    for i, j in ti.ndrange(NV, NV):
        K[i, j] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])


# compute constraint vector and gradient vector
@ti.kernel
def computeCg():
    for i in range(NE):
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
    dim = (2 * NV + NE)  # the system dimension

    A = np.zeros((dim, dim), dtype=np.float64)
    # uppper left: mass matrix
    for i in range(NV):
        A[2 * i, 2 * i] = mass[i]
        A[2 * i + 1, 2 * i + 1] = mass[i]

    # uppper left: geometric stiffness
    if step != 0:
        for i in range(NV):
            for j in range(NV):
                A[2 * i:2 * i + 2, 2 * j:2 * j + 2] -= KK[i, j]
    # print("K matrix: \n", repr(KK))
    # Other parts
    start = 2 * NV
    for i in range(NE):
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
    G = np.zeros((2 * NV, NE))
    for i in range(NE):
        idx1, idx2 = cidx[i]
        g0 = g[2 * i + 0]
        g1 = g[2 * i + 1]
        G[2 * idx1:2 * idx1 + 2, i] = g0
        G[2 * idx2:2 * idx2 + 2, i] = g1
    Gl = G @ l
    # RHS
    b = np.zeros(dim, dtype=np.float64)
    b[2 * NV:] = -c
    # geometric stiffness
    for i in range(1, NV):
        b[2 * i:2 * i + 2] = -mass[i] * (p[i] - prep[i])
    b[:2 * NV] += Gl
    # np.set_printoptions(precision=5, suppress=False)
    # print("b: ", b)
    x = np.linalg.solve(A[2:, 2:], b[2:])
    return x


@ti.kernel
def updateV():
    for i in range(NV):
        if mass[i] != 0.0:
            vel[i] = (pos[i] - oldPos[i]) / h


@ti.kernel
def computeResidual(step: ti.i32):
    dualResidual[None] = 0.0
    primalResidual[None] = 0.0
    for i in range(NE):
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
    for i in range(NV):
        dx = tpos[i] - predictionPos[i]
        e += mass[i] * dx.norm_sqr()
    for i in range(NE):
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
    for i in range(NV - 1):
        tpos[i +
             1] = pos[i +
                      1] + stepsize * ti.Vector([dx[2 * i + 0], dx[2 * i + 1]])
    for i in range(NE):
        tlag[i] = lagrangian[i] + stepsize * dx[2 * (NV - 1) + i]


@ti.kernel
def updatePosLambda():
    for i in range(NV):
        pos[i] = tpos[i]
    for i in range(NE):
        lagrangian[i] = tlag[i]


init_pos()
init_mesh()
initConstraint()
gui = ti.GUI('Mesh Stable Constrainted Dynamics')
pause = True
while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        if e.key == gui.SPACE:
            pause = not pause
    mouse_pos = gui.get_cursor_pos()
    attractor_pos[None] = mouse_pos
    attractor_strength[None] = gui.is_pressed(gui.LMB) - gui.is_pressed(
        gui.RMB)
    gui.circle(mouse_pos, radius=15, color=0x336699)
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
                setTposTlag(1.0, dx)
                updatePosLambda()
            computeResidual(step)
            updateV()
        # print("max dual residual: ", maxdualResidual.to_numpy())
        # print("max prim residual: ", maxprimalResidual.to_numpy())

    position = pos.to_numpy()
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
#             np.savetxt('data/lambda.txt',np.reshape(lagrangian.to_numpy(),(1,NE)),delimiter=',')

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
#         print("Sum lambda: ", repr(np.reshape(lagrangian.to_numpy(),(1,NE))))

#         if i == 20 and ite == 10:
#             np.savetxt('data/updatedPos.txt', np.reshape(pos.to_numpy(),(1,2*N)), delimiter=',')
#             np.savetxt('data/updateLambda.txt',np.reshape(lagrangian.to_numpy(),(1,NE)),delimiter=',')

#         # print("Lagrangian Multipler: ", lagrangian.to_numpy())
#         # print(f"\n-----------------------------end iteration  {ite}-------------------------------")
#     print("Old position:", repr(np.reshape(oldPos.to_numpy(),(1,2*N))))
#     updateV()
#     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Velocities: ", repr(np.reshape(vel.to_numpy(),(1,2*N))))
