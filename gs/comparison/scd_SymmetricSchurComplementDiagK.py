"""
Rod simulation based on [Stable Constrainted Dynamics, Maxime Tournier et.al, 2015.] 
with Schur complement and we make the global system matrix symmetric.
"""
import taichi as ti
from taichi.lang.ops import abs, sqrt
import numpy as np
from numpy.linalg import inv

ti.init(arch=ti.cpu)
gravity = ti.Vector([0, -9.8])
h = 0.01  # timestep size

NStep = 1  # number of steps in each frame
NMaxIte = 5  # number of iterations in each step
N = 5  # number of particles
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
    mass[N-1] = 100.0


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
        constraint[i] = l - rest_len - alpha * lagrangian[i]
        # print("p1: ",p1, ", p2: ", p2)
        # print("lambda: ", lagrangian[i])
        # print("constraint ", i , ": ", constraint[i])
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

    # uppper left: geometric stiffness
    for i in range(N):
        A[2*i, 2*i]    += KK[i,i][0,0]
        A[2*i+1,2*i+1] += KK[i,i][1,1]
    # for i in range(N):
    #     for j in range(N):
    #         A[2 * i:2 * i + 2, 2 * j:2 * j + 2] += KK[i, j]

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
    print(f"norm(M(x-y)): {np.linalg.norm(u[2:])}")
    Gl = G @ l
    print(f"norm(GL): {np.linalg.norm(Gl[2:])}")
    u -= Gl
    print(f">>> Primal Residual: {np.linalg.norm(u[2:])}")
    v = -c
    print(f">>> Dual Residual: {np.linalg.norm(v)}")
    # print(f"c: {c}")
    # static point removed
    A = A[2:, 2:]
    G = G[2:, :]
    u = u[2:]
    # print(f"A: \n{A}")
    # print(f"G: \n{G}")
    # print(f"u: {u}, v: {v}")
    # Schur complement
    GTAinv = np.transpose(G) @ inv(A)
    S = complianceMatrix - GTAinv @ G

    b = v - GTAinv @ u
    y = np.linalg.solve(S, b)  # delta lambda ??? some problems happend here
    x = np.linalg.solve(A, u - G @ y)  # delta pos
    # print(f"dx: {x}")
    # print(f"dl: {y}")
    print(f"norm(dx): {np.linalg.norm(x)}")
    print(f"norm(dl): {np.linalg.norm(y)}")
    return x, y


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
        mass1 = mass[idx1]
        mass2 = mass[idx2]
        p1, p2 = pos[idx1], pos[idx2]
        constraint = (p1 - p2).norm() - rest_len

        dualResidual[None] += abs(constraint + alpha * lagrangian[i])

        gradient = (p1 - p2).normalized()
        r0 = ti.Vector([0.0, 0.0])
        r1 = r0
        if mass1 != 0.0:
            r0 = mass1 * (
                p1 - predictionPos[idx1]) + lagrangian[i] * gradient
        if mass2 != 0.0:
            r1 = mass2 * (
                p2 - predictionPos[idx2]) - lagrangian[i] * gradient
        primalResidual[None] += sqrt(r0.norm_sqr() + r1.norm_sqr())
    if maxdualResidual[None] < dualResidual[None]:
        maxdualResidual[None] = dualResidual[None]
    if maxprimalResidual[None] < primalResidual[None]:
        maxprimalResidual[None] = primalResidual[None]
    print("-----iteration:", step, "-------------")
    print("Dual Residual: ", dualResidual[None])
    print("Primal Residual: ", primalResidual[None])


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
            print(f"######################### Timestep: {count} ##############################")
            count += 1
            semiEuler()
            for ite in range(NMaxIte):
                resetK()
                computeCg()
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

    filename = f'./data/frame_{frame:05d}.png'   # create filename with suffix png
    frame += 1
    if frame == 300:
        break
    gui.show(filename)
    # gui.show()