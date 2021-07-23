"""
Rod simulation based on [Stable Constrainted Dynamics, Maxime Tournier et.al, 2015.] 
with Schur complement and Symmetric global system matrix.

The schur complement system matrix is solved by two CR direct solver.

The Geometric stiffness matrix is approximated by two previous step gradient.
"""
from numpy.lib import RankWarning
import taichi as ti
from taichi.lang.ops import max
import numpy as np
from numpy.linalg import inv

ti.init(arch=ti.cpu)
gravity = ti.Vector([0, -9.8])
h = 0.01  # timestep size

NStep = 1  # number of steps in each frame
NMaxIte = 5  # number of iterations in each step
N = 200  # number of particles
NC = N - 1  # number of distance constraint
# CR
MaxCRIte = 10
LastMass = 100.0

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


# g in Non-smooth Newton methods for deformable multi-body dynamics, Eq.57
g_pre = ti.Vector.field(2, ti.f64, N)
g = ti.Vector.field(2, ti.f64, N)

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
    mass[N-1] = LastMass


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
def computeCg(dx: ti.ext_arr()):
    for i in range(NC):
        idx1, idx2 = disConsIdx[i]
        rest_len = disConsLen[i]
        mass1 = mass[idx1]
        mass2 = mass[idx2]
        sumInvMass = mass1 + mass2
        if sumInvMass < 1.0e-6:
            print("Wrong Mass Setting")
        p1, p2 = pos[idx1], pos[idx2]
        l = (p1 - p2).norm()
        n = (p1 - p2).normalized()
        # xpbd
        constraint[i] = l - rest_len - alpha * lagrangian[i]
        # print("p1: ",p1, ", p2: ", p2)
        # print("lambda: ", lagrangian[i])
        # print("constraint ", i , ": ", constraint[i])
        gradient[2 * i + 0] = n
        gradient[2 * i + 1] = -n
    # Approximated Geometric Stifness Matrix
    # please see: Non-Smooth Newton Methods for Deformable Multi-Body Dynamics, Eq.73
    for i in range(N):
        if mass[i] != 0.0 and dx[2*i-2] !=0.0 and dx[2*i-1] !=0.0:
            dg1 = g[i] - g_pre[i]
            K[i,i][0,0] += max(0,dg1[0]/dx[2*i-2] + mass[i])
            K[i,i][1,1] += max(0,dg1[1]/dx[2*i-1] + mass[i])


"""
Conjugate residual method
"""
def CR(A, b, x0):
    m, n = A.shape
    assert (m == n and m == b.shape[0] and b.shape == x0.shape)
    residual = 1.0e-6
    r0 = b - A @ x0
    if np.linalg.norm(r0) < residual:
        return x0
    x = x0
    p = r0
    r = r0
    count = 0
    Ap = A @ p
    while True:
        Ar = A @ r
        rAr = sum(r * Ar)
        alpha = rAr / sum(Ap * Ap)
        x = x + alpha * p
        r_next = r - alpha * Ap
        if np.linalg.norm(r_next) < residual:
            break
        Ar_next = A @ r_next
        beta = sum(r_next * Ar_next) / rAr
        p = r_next + beta * p
        r = r_next
        Ap = Ar_next + beta * Ap
        count += 1
    return x


def computeSv(complianceMatrix, G, A, v):
    Gv = G @ v
    x0 = np.zeros(2 * (N - 1), dtype=np.float64)
    AinvGV = CR(A, Gv, x0)
    return complianceMatrix @ v - np.transpose(G) @ AinvGV


"""
Two CR Iteration to solve the big linear system
"""


def CRwithCR(complianceMatrix, G, A, b, x0):
    m, n = A.shape
    assert (m == n and b.shape == x0.shape)
    residual = 1.0e-6
    Sx0 = computeSv(complianceMatrix, G, A, x0)
    r0 = b - Sx0
    if np.linalg.norm(r0) < residual:
        return x0
    x = x0
    p = r0
    r = r0
    count = 0
    Sp = computeSv(complianceMatrix, G, A, p)
    while count < MaxCRIte:
        Sr = computeSv(complianceMatrix, G, A, r)
        rSr = sum(r * Sr)
        alpha = rSr / sum(Sp * Sp)
        x = x + alpha * p
        r_next = r - alpha * Sp
        if np.linalg.norm(r_next) < residual:
            break
        Sr_next = computeSv(complianceMatrix, G, A, r_next)
        beta = sum(r_next * Sr_next) / rSr
        p = r_next + beta * p
        r = r_next
        Sp = Sr_next + beta * Sp
        count += 1
    # print(f"number of iteration: {count}")
    return x

@ti.kernel
def update_g(g_new: ti.ext_arr()):
    for i in range(N):
        g_pre[i] = g[i]
        g[i]  = ti.Vector([g_new[2*i + 0], g_new[2 * i +1]])

"""
Solve the linear system with schur complement method
A x = b
A = | M+K    G  |
    | G'  -alpha|
x = | x |
    | y |
b = | u | = |      -M(x - y) - J' * lambda      |
    | v |   | -(constraint - alpha * lambda)    |

Reference: https://en.wikipedia.org/wiki/Schur_complement
"""


def solveWithSchurComplement(mass, p, prep, g, KK, l, c, cidx, iteration):
    print(f"------------------  iteration: {iteration} --------------")
    dim = 2 * N
    # upper left matrix
    A = np.zeros((dim, dim), dtype=np.float64)
    # uppper left: mass matrix
    for i in range(N):
        A[2 * i, 2 * i] = mass[i]
        A[2 * i + 1, 2 * i + 1] = mass[i]

    # uppper left:diagnoal geometric stiffness
    for i in range(N):
        A[2 * i, 2 * i] += KK[i, i][0, 0]
        A[2 * i + 1, 2 * i + 1] += KK[i, i][1, 1]

    # print(f"A.shape: {A.shape}")
    # print(f"A: {A}")
    # gradient matrix
    G = np.zeros((2 * N, NC))
    for i in range(NC):
        idx1, idx2 = cidx[i]
        g0 = g[2 * i + 0]
        g1 = g[2 * i + 1]
        G[2 * idx1:2 * idx1 + 2, i] = g0
        G[2 * idx2:2 * idx2 + 2, i] = g1

    # compliance matrix
    complianceMatrix = np.zeros((NC, NC), dtype=np.float64)
    np.fill_diagonal(complianceMatrix, -alpha)

    # RHS
    u = np.zeros(dim, dtype=np.float64)
    # geometric stiffness
    for i in range(1, N):
        u[2 * i:2 * i + 2] = -mass[i] * (p[i] - prep[i])
    np.set_printoptions(precision=5, suppress=False)
    print(f"nomr(lambda): {np.linalg.norm(l)}")
    # print(f"norm(M(x-y)): {np.linalg.norm(u[2:])}")
    Gl = G @ l
    # print(f"norm(GL): {np.linalg.norm(Gl[2:])}")
    # print(f"{l} <<< Lagrangian")
    u -= Gl
    
    # update g 
    update_g(u)

    print(f">>> Primal Residual: {np.linalg.norm(u[2:])}")
    v = -c
    print(f">>> Dual Residual: {np.linalg.norm(v)}")
    # print(f"c: {c}")
    # static point removed
    A = A[2:, 2:]
    G = G[2:, :]
    u = u[2:]

    # CG Solver
    # print(f"A: \n {A}")
    x0 = np.zeros(2 * (N - 1), dtype=np.float64)
    AinvU = CR(A, u, x0)
    b = v - np.transpose(G) @ AinvU
    # Big CG Solver
    x0 = np.zeros(NC, dtype=np.float64)
    print(f"A.shape:{A.shape}")
    print(f"b.shape:{b.shape}")
    print(f"x0.shape:{x0.shape}")
    dl = CRwithCR(complianceMatrix, G, A, b, x0)
    dx = np.linalg.solve(A, u - G @ dl)

    print(f"norm(dx): {np.linalg.norm(dx)}")
    print(f"norm(dl): {np.linalg.norm(dl)}")
    return dx, dl


@ti.kernel
def updateV():
    for i in range(N):
        if mass[i] != 0.0:
            vel[i] = (pos[i] - oldPos[i]) / h


@ti.kernel
def updatePosLambda(dx: ti.ext_arr(), dl: ti.ext_arr()):
    for i in range(N - 1):
        pos[i + 1] += ti.Vector([dx[2 * i + 0], dx[2 * i + 1]])
    for i in range(NC):
        lagrangian[i] += dl[i]


initRod()
initConstraint()
gui = ti.GUI('Stable Constrainted Dynamics')
pause = False
count = 0
frame = 0
while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        if e.key == gui.SPACE:
            pause = not pause
    if not pause:
        for step in range(NStep):
            print(
                f"######################### Timestep: {count} ##############################"
            )
            count += 1
            semiEuler()
            dx = np.zeros(2 * (N - 1), dtype=np.float64)
            for ite in range(NMaxIte):
                resetK()
                computeCg(dx)
                dx, dl = solveWithSchurComplement(mass.to_numpy(),
                                                  pos.to_numpy(),
                                                  predictionPos.to_numpy(),
                                                  gradient.to_numpy(),
                                                  K.to_numpy(),
                                                  lagrangian.to_numpy(),
                                                  constraint.to_numpy(),
                                                  disConsIdx.to_numpy(), ite)
                updatePosLambda(dx, dl)
            updateV()

    position = pos.to_numpy()
    begin = position[:-1]
    end = position[1:]
    gui.lines(begin, end, radius=3, color=0x0000FF)
    gui.circles(pos.to_numpy(), radius=5, color=0xffaa33)
    filename = f'./data/frame_{frame:05d}.png'  # create filename with suffix png
    frame += 1
    if frame == 300:
        break
    gui.show(filename)
