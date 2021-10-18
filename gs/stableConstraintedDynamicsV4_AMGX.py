"""
Rod simulation based on [Stable Constrainted Dynamics, Maxime Tournier et.al, 2015.]
"""
import taichi as ti
from taichi.lang.ops import abs, sqrt
import numpy as np

#------------------------------Begin AMGX Solver--------------------------------
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import pyamgx

pyamgx.initialize()

# Initialize config and resources:
cfg = pyamgx.Config().create_from_dict({
     "config_version": 2,
     "determinism_flag": 1,
     "exception_handling" : 1,
     "solver": {
         "monitor_residual": 1,
         "solver": "BICGSTAB",
         "convergence": "RELATIVE_INI_CORE",
         "preconditioner": {
             "solver": "NOSOLVER"
         }
     }
 })

rsc = pyamgx.Resources().create_simple(cfg)

# Create matrices and vectors:
A_amgx = pyamgx.Matrix().create(rsc)
b_amgx = pyamgx.Vector().create(rsc)
x_amgx = pyamgx.Vector().create(rsc)

# Create solver:
solver_amgx = pyamgx.Solver().create(rsc, cfg)
#------------------------------End AMGX Solver--------------------------------

ti.init(arch=ti.cpu)
gravity = ti.Vector([0, -9.8])
h = 0.01  # timestep size

NStep = 1  # number of steps in each frame
NMaxIte = 3  # number of iterations in each step
N = 3  # number of particles
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
    # mass[9] = 100.0


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
Assemble system matrix
A =   |  M - K     -J' |
      |  J       alpha |
b = |      -M(x - y) + J' * lambda      |
    | -(constraint + alpha * lambda)    |
"""


def assemble(mass, p, prep, g, KK, l, c, cidx, iteration):
    print(f"------------------  iteration: {iteration} --------------")
    dim = (2 * N + NC)  # the system dimension

    A = np.zeros((dim, dim), dtype=np.float64)
    # uppper left: mass matrix
    for i in range(N):
        A[2 * i, 2 * i] = mass[i]
        A[2 * i + 1, 2 * i + 1] = mass[i]

    # uppper left: geometric stiffness
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
    np.set_printoptions(precision=5, suppress=False)
    # geometric stiffness
    for i in range(1, N):
        b[2 * i:2 * i + 2] = -mass[i] * (p[i] - prep[i])
    print(f"norm(lambda): {np.linalg.norm(l)} ")
    print(f"norm(M(x-y)): {np.linalg.norm(b[2:2*N])} ")
    print(f"norm(GL)    : {np.linalg.norm(Gl[2:2*N])} ")
    b[:2 * N] += Gl
    print(f">>> Primal Residual: {np.linalg.norm(b[2:2*N])}")
    print(f">>> Dual Residual: {np.linalg.norm(b[2*N:])}")
    # print(f"c: {c}")
    x = np.linalg.solve(A[2:, 2:], b[2:])

    #------------ Begin AMGX Solver----------------
    A_global_sparse = sparse.csr_matrix(A[2:, 2:])
    A_amgx.upload_CSR(A_global_sparse)
    b_amgx.upload(b[2:])
    solusion = np.zeros(dim-2, dtype=np.float64)
    x_amgx.upload(solusion)
    solver_amgx.setup(A_amgx)
    solver_amgx.solve(b_amgx, x_amgx)
    x_amgx.download(solusion)
    print("pyamgx solution: ", solusion)
    print("scipy solution: ", splinalg.spsolve(A_global_sparse, b[2:]))
    print("Correct Solusion distance: ", np.linalg.norm(solusion-x))
    


    # print(f"A: \n{A[2:2*N,2:2*N]}")
    # print(f"G: \n{G[2:,:]}")
    # print(f"dx: {x[:2*(N-1)]}")
    # print(f"dl: {x[2*(N-1):]}")
    # print(f"[u, v]     : {b} ")
    print(f"norm(dx): {np.linalg.norm(x[:2*(N-1)])}")
    print(f"norm(dl): {np.linalg.norm(x[2*(N-1):])}")
    return solusion


@ti.kernel
def updateV():
    for i in range(N):
        if mass[i] != 0.0:
            vel[i] = (pos[i] - oldPos[i]) / h


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
def updatePosLambda(dx: ti.ext_arr()):
    for i in range(N - 1):
        pos[i + 1] += ti.Vector([dx[2 * i + 0], dx[2 * i + 1]])
    for i in range(NC):
        lagrangian[i] += dx[2*(N-1)+i]


initRod()
initConstraint()
gui = ti.GUI('Stable Constrainted Dynamics v3')
pause = True
count = 0
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
                dx = assemble(mass.to_numpy(), pos.to_numpy(),
                              predictionPos.to_numpy(), gradient.to_numpy(),
                              K.to_numpy(), lagrangian.to_numpy(),
                              constraint.to_numpy(), disConsIdx.to_numpy(),
                              ite)
                updatePosLambda(dx)
            updateV()

    position = pos.to_numpy()
    begin = position[:-1]
    end = position[1:]
    gui.lines(begin, end, radius=3, color=0x0000FF)
    gui.circles(pos.to_numpy(), radius=5, color=0xffaa33)
    gui.show()

# for step in range(NStep):
#     print(f"######################### Timestep: {step} ##############################")
#     semiEuler()
#     for ite in range(NMaxIte):
#         resetK()
#         computeCg()
#         dx = assemble(mass.to_numpy(), pos.to_numpy(),
#                       predictionPos.to_numpy(), gradient.to_numpy(),
#                       K.to_numpy(), lagrangian.to_numpy(),
#                       constraint.to_numpy(), disConsIdx.to_numpy(),
#                       ite)
#         updatePosLambda(dx)
#     updateV()