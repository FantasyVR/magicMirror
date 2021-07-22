"""
Mass-spring system simulation based on [Stable Constrainted Dynamics, Maxime Tournier et.al, 2015.]

Solve the symmetric KKT system with Schur complement method and 2 CG solver
"""
import taichi as ti
from taichi.lang.ops import abs, sqrt
import numpy as np
from numpy.linalg import inv

ti.init(arch=ti.cpu)
gravity = ti.Vector([0, -9.8])
h = 0.01  # timestep size

NStep = 5  # number of steps in each frame
NMaxIte = 1  # number of iterations in each step
N = 20  # number of particles
NE = (2 * N + 1) * N * 2  # number of edges
NV = (N + 1)**2  # number of vertices
# CG
MaxCGIte = 10

pos = ti.Vector.field(2, ti.f64, NV)
oldPos = ti.Vector.field(2, ti.f64, NV)
predictionPos = ti.Vector.field(2, ti.f64, NV)
vel = ti.Vector.field(2, ti.f64, NV)
mass = ti.field(ti.f64, NV)

disConsIdx = ti.Vector.field(
    2, int, NE)  # each element store vertex indices of the constraint
disConsLen = ti.field(
    ti.f64, NE
)  # rest state (rest length of spring in this example) of each constraint
gradient = ti.Vector.field(2, ti.f64, 2 * NE)  # gradient of constraints
constraint = ti.field(ti.f64, NE)  # constraints violation

#xpbd values
compliance = 1.0e-6
alpha = compliance / h / h
lagrangian = ti.field(ti.f64, NE)

# geometric stiffness
K = ti.Matrix.field(2, 2, ti.f64, (NV, NV))

# For validation
dualResidual = ti.field(ti.f64, ())
primalResidual = ti.field(ti.f64, ())
maxdualResidual = ti.field(ti.f64, ())
maxprimalResidual = ti.field(ti.f64, ())

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
    for i, j in ti.ndrange(N + 1, N):
        # horizontal
        a = i * (N + 1) + j
        disConsIdx[i * N + j] = ti.Vector([a, a + 1])
    start = N * (N + 1)
    for i, j in ti.ndrange(N, N + 1):
        # vertical
        a = i * (N + 1) + j
        disConsIdx[start + i + j * N] = ti.Vector([a, a + N + 1])
    start = 2 * start
    for i, j in ti.ndrange(N, N):
        # diagonal
        a = i * (N + 1) + j
        disConsIdx[start + i * N + j] = ti.Vector([a, a + N + 2])
    start += N * N
    for i, j in ti.ndrange(N, N):
        a = i * (N + 1) + j
        disConsIdx[start + i * N + j] = ti.Vector([a + 1, a + N + 1])


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
        constraint[i] = l - rest_len - alpha * lagrangian[i]
        gradient[2 * i + 0] = n
        gradient[2 * i + 1] = -n
        # geometric stiffness
        """
            k = lambda[i]/l * (I - n * n')
            K = | Hessian_{x1,x1}, Hessian_{x1,x2}   |  = | k  -k|
                | Hessian_{x1,x2}, Hessian_{x2,x2}   |    |-k   k|
        """
        if lagrangian[i] > 0.0:
            I = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
            k = lagrangian[i] / l * (I - n @ n.transpose())
            K[idx1, idx1] += k
            K[idx1, idx2] -= k
            K[idx2, idx1] -= k
            K[idx2, idx2] += k


def computeSv(complianceMatrix, G, A, v):
    Gv = G @ v
    x0 = np.zeros(2*(NV-1),dtype=np.float64)
    AinvGV = CG(A, Gv, x0)
    return complianceMatrix @ v - np.transpose(G) @ AinvGV

"""
Conjugate Gradient Method with LLT direct Solver
"""
"""
Conjugate gradient solver
"""
def CG(A, b, x0):
    m, n = A.shape
    assert (m == n and b.shape == x0.shape)
    residual = 1.0e-6
    r0 = b - A @ x0
    if np.linalg.norm(r0) < residual:
        return x0
    x = x0
    p = r0
    r = r0
    count = 0
    while count < MaxCGIte:
        Ap = A @ p
        rr = sum(r * r)
        alpha = rr / sum(p * Ap)
        x = x + alpha * p
        r_next = r - alpha * Ap
        if np.linalg.norm(r_next) < residual:
            break
        beta = sum(r_next * r_next) / rr
        p = r_next + beta * p
        r = r_next
        count += 1
    # print(f"number of iteration: {count}")
    return x
"""
Two CG Iteration to solve the big linear system
"""
def CGwithCG(complianceMatrix, G, A, b, x0):
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
    while count < MaxCGIte:
        Sp = computeSv(complianceMatrix, G, A, p)
        rr = sum(r * r)
        alpha = rr / sum(p * Sp)
        x = x + alpha * p
        r_next = r - alpha * Sp
        if np.linalg.norm(r_next) < residual:
            break
        beta = sum(r_next * r_next) / rr
        p = r_next + beta * p
        r = r_next
        count += 1
    # print(f"number of iteration: {count}")
    return x

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


def solveWithSchurComplement(mass, p, prep, g, KK, l, c, cidx, step):
    dim = 2 * NV  # the system dimension

    A = np.zeros((dim, dim), dtype=np.float64)
    # uppper left: mass matrix
    for i in range(NV):
        A[2 * i, 2 * i] = mass[i]
        A[2 * i + 1, 2 * i + 1] = mass[i]

    # uppper left: geometric stiffness
    if step != 0:
        for i in range(NV):
            for j in range(NV):
                A[2 * i:2 * i + 2, 2 * j:2 * j + 2] += KK[i, j]

    # gradient matrix
    G = np.zeros((2 * NV, NE))
    for i in range(NE):
        idx1, idx2 = cidx[i]
        g0 = g[2 * i + 0]
        g1 = g[2 * i + 1]
        G[2 * idx1:2 * idx1 + 2, i] = g0
        G[2 * idx2:2 * idx2 + 2, i] = g1

    # compliance matrix
    complianceMatrix = np.zeros((NE, NE), dtype=np.float64)
    np.fill_diagonal(complianceMatrix, -alpha)

    # RHS
    u = np.zeros(dim, dtype=np.float64)
    for i in range(1, NV):
        u[2 * i:2 * i + 2] = -mass[i] * (p[i] - prep[i])
    np.set_printoptions(precision=5, suppress=False)
    print(f"nomr(lambda): {np.linalg.norm(l)}")
    Gl = G @ l
    u -= Gl
    print(f">>> Primal Residual: {np.linalg.norm(u[2:])}")
    v = -c
    print(f">>> Dual Residual: {np.linalg.norm(v)}")

    A = A[2:, 2:]
    G = G[2:, :]
    u = u[2:]

    # CG Solver
    x0 = np.zeros(2*(NV-1),dtype=np.float64)
    AinvU = CG(A, u, x0)
    b = v - np.transpose(G) @ AinvU

    x0 = np.zeros(NE, dtype=np.float64)
    dl = CGwithCG(complianceMatrix, G, A, b, x0)
    dx = np.linalg.solve(A, u - G @ dl)
    print(f"norm(dx): {np.linalg.norm(dx)}")
    print(f"norm(dl): {np.linalg.norm(dl)}")
    return dx, dl


@ti.kernel
def updateV():
    for i in range(NV):
        if mass[i] != 0.0:
            vel[i] = (pos[i] - oldPos[i]) / h


@ti.kernel
def updatePosLambda(dx: ti.ext_arr(), dl: ti.ext_arr()):
    for i in range(NV - 1):
        pos[i + 1] += ti.Vector([dx[2 * i + 0], dx[2 * i + 1]])
    for i in range(NE):
        lagrangian[i] += dl[i]


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
        for step in range(NStep):
            semiEuler()
            for ite in range(NMaxIte):
                resetK()
                computeCg()
                dx, dl = solveWithSchurComplement(mass.to_numpy(), pos.to_numpy(),
                                  predictionPos.to_numpy(),
                                  gradient.to_numpy(), K.to_numpy(),
                                  lagrangian.to_numpy(), constraint.to_numpy(),
                                  disConsIdx.to_numpy(), step)
                updatePosLambda(dx, dl)
            updateV()

    position = pos.to_numpy()
    gui.circles(pos.to_numpy(), radius=5, color=0xffaa33)
    gui.show()
