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
from taichi.core.record import start_recording
from taichi.lang.ops import abs, sqrt
import numpy as np
import scipy as sp
from scipy.linalg import ldl

ti.init(arch=ti.gpu)
# general setting
gravity = ti.Vector([0, -0.98])
h = 0.01  # timestep size
squareInverseTime = 1.0 / (h * h)
compliance = 1.0e-6
alpha = compliance * squareInverseTime
alpha = 0.0
N = 3  # number of particles
NC = N - 1  # number of distance constraint
NStep = 1
NMaxIte = 1
omega = 1.0  # Relaxtion ratio

pos             = ti.Vector.field(2, float, N)
oldPos          = ti.Vector.field(2, float, N)
predictionPos   = ti.Vector.field(2, float, N)
vel             = ti.Vector.field(2, float, N)
invmass         = ti.field(float, N)

disConsIdx  = ti.Vector.field(2, int, NC)
disConsLen  = ti.field(float, NC)
lagrangian  = ti.field(float, NC)
gradient    = ti.Vector.field(2, float, 2 * NC)
K           = ti.Matrix.field(2, 2, float, (N, N))
constraint  = ti.field(float, NC)
# For validation
dualResidual    = ti.field(float, ())
primalResidual   = ti.field(float, ())


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
            vel[i] = vel[i] +  h * gravity
            oldPos[i] = pos[i]
            pos[i] =  pos[i] + h * vel[i]
            predictionPos[i] = pos[i]


@ti.kernel
def resetLagrangin():
    for i in range(NC):
        lagrangian[i] = 0.0


# compute constraint vector and gradient vector and hessian matrix K
@ti.kernel
def computeCgK():
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
        constraint[i] = l - rest_len + alpha * lagrangian[i]
        gradient[2 * i + 0] = (p1 - p2).normalized()
        gradient[2 * i + 1] = -gradient[2 * i + 0]
        # compute geometric stiffness
        I = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
        k = I - 1.0 / (l**2) * (p1 - p2) @ (p1 - p2).transpose()
        k *= -lagrangian[i] / l
        K[idx1, idx1] += -k
        K[idx1, idx2] += k
        K[idx2, idx1] += -k
        K[idx2, idx2] += k

def assemble(mass, g, l, c, cidx, oldP, predictP, K):
    dim = (2 * N + NC)  # the system dimension

    A = np.zeros((dim, dim), dtype=np.float64)
    b = np.zeros(dim, dtype=np.float64)
    # uppper left: mass matrix + geometric stiffness matrix
    for i in range(N):
        A[2 * i, 2 * i] = mass[i]
        A[2 * i + 1, 2 * i + 1] = mass[i]
    # print(A)
    # geometric stiffness matrix
    # for i in range(N):
    #     for j in range(N):
    #         A[2 * i:2 * i + 2, 2 * j:2 * j + 2] = K[i, j]

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
        A[2 * idx1:2 * idx1 + 2, start + i] -= g0
        A[2 * idx2:2 * idx2 + 2, start + i] -= g1
        # lower right
        A[start + i, start + i] = -alpha
    
    np.set_printoptions(precision=2,suppress = True)
    print(A)

    G = np.zeros((2 * N, NC))
    for i in range(NC):
        idx1, idx2 = cidx[i]
        g0 = g[2 * i + 0]
        g1 = g[2 * i + 1]
        G[2 * idx1:2 * idx1 + 2, i] += g0
        G[2 * idx2:2 * idx2 + 2, i] += g1
    Glambda = G @ l

    # compute the rank of gradient matrix
    print("rank(G) = ", np.linalg.matrix_rank(G),"; real rank should be: ", NC)
    print("rank(A) = ", np.linalg.matrix_rank(A))
    # b_uppper: -M(x^n - \\tilta{x}) + \\nabla C(x^n) * \lambda
    # for i in range(N):
    #     b[2 * i:2 * i +
    #       2] = -mass[i] * (oldP[i] - predictP[i]) + Glambda[2 * i:2 * i + 2]
    # b_lower: C(x^n)+ \\tilta {\\alpha} * \lambda, we could use numpy broad cast
    # b[2 * N:] = c + alpha * l
    b[2 * N:] = -c 

    # Cholesky factorization of A = LDLT, A could be SPD, SemiPD, NPD
    # Ly = b -> DLT x = y
    # x = np.linalg.solve(A[2:, 2:], b[2:])
    print(f"p1: {predictP[0]}, p2: {predictP[1]}")
    print(f"p1: {predictP[1]}, p2: {predictP[2]}")
    print(np.allclose(predictP[0]-predictP[1], np.zeros(2)))
    print(np.allclose(predictP[1]-predictP[2], np.zeros(2)))
    np.set_printoptions(precision=2, suppress = True)
    print(f"Real system matrix: \n {A[2:,2:]} \n\n-----------------------")
    lu, d, perm = ldl(A[2:,2:])
    y = np.linalg.solve(lu[perm, :], b[2:])
    x = np.linalg.solve(d.dot(lu[perm, :].T), y)
    return x
 

@ti.kernel
def updatePosLag(x: ti.ext_arr()):
    for i in range(N):
        if invmass[i] != 0.0:
            pos[i] = pos[i] + ti.Vector([x[2 * i + 0], x[2 * i + 1]])
    for i in range(NC):
        lagrangian[i] = lagrangian[i] + x[2 * N + i]


@ti.kernel
def updateV():
    for i in range(N):
        vel[i] = (pos[i] - oldPos[i]) / h


initRod()
initConstraint()
gui = ti.GUI('XPBD with Global System')
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
                    computeCgK()
                    x = assemble(invmass.to_numpy(), gradient.to_numpy(),
                                lagrangian.to_numpy(), constraint.to_numpy(),
                                disConsIdx.to_numpy(), oldPos.to_numpy(),
                                pos.to_numpy(), K.to_numpy())
                    updatePosLag(x)
                updateV()
    position = pos.to_numpy()
    begin = position[:-1]
    end = position[1:]
    gui.lines(begin, end, radius=3, color=0x0000FF)
    gui.circles(pos.to_numpy(), radius=5, color=0xffaa33)
    gui.show()
