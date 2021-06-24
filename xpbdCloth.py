"""
We use XPBD-FEM (cpu version) to simulate the deformation of 2D object
"""
import taichi as ti
from taichi.lang.ops import sqrt

ti.init(arch=ti.cpu, kernel_profiler=True) # must be cpu

h = 0.1  # timestep size

compliance = 1.0e-3  # Fat Tissuse compliance, for more specific material,please see: http://blog.mmacklin.com/2016/10/12/xpbd-slides-and-stiffness/
alpha = compliance * (1.0 / h / h
                      )  # timestep related compliance, see XPBD paper
N = 50
NF = 2 * N**2  # number of faces
NV = (N + 1)**2  # number of vertices
pos = ti.Vector.field(2, float, NV)
oldPos = ti.Vector.field(2, float, NV)
vel = ti.Vector.field(2, float, NV)  # velocity of particles
invMass = ti.field(float, NV)  #inverse mass of particles
f2v = ti.Vector.field(3, int, NF)  # ids of three vertices of each face
B = ti.Matrix.field(2, 2, float, NF)  # D_m^{-1}
F = ti.Matrix.field(2, 2, float, NF)  # deformation gradient
lagrangian = ti.field(float, NF)  # lagrangian multipliers
gravity = ti.Vector([0, -1.2])
MaxIte = 5
NumSteps = 3

attractor_pos = ti.Vector.field(2, float, ())
attractor_strength = ti.field(float, ())


@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, N + 1):
        k = i * (N + 1) + j
        pos[k] = ti.Vector([i, j]) / N * 0.5 + ti.Vector([0.25, 0.25])
        oldPos[k] = pos[k]
        vel[k] = ti.Vector([0, 0])
        invMass[k] = 1.0
    # for i in range(N + 1):
    #     k = i * (N + 1) + N
    #     invMass[k] = 0.0
    k0 = N
    k1 = (N + 2) * N
    invMass[k0] = 0.0
    invMass[k1] = 0.0
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        B_i_inv = ti.Matrix.cols([b - a, c - a])
        B[i] = B_i_inv.inverse()


@ti.kernel
def init_mesh():
    for i, j in ti.ndrange(N, N):
        k = (i * N + j) * 2
        a = i * (N + 1) + j
        b = a + 1
        c = a + N + 2
        d = a + N + 1
        f2v[k + 0] = [a, b, c]
        f2v[k + 1] = [c, d, a]


@ti.func
def resetLagrangian():
    for i in range(NF):
        lagrangian[i] = 0.0


@ti.func
def computeGradient(idx, U, S, V):
    sumSigma = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2)
    #print("Befor Sumsigma", sumSigma)
    if sumSigma < 0.0000001:
        sumSigma = 1.0
    #print("SumSigma: ", sumSigma)
    dcdS = 1.0 / sumSigma * ti.Vector([S[0, 0] - 1, S[1, 1] - 1
                                       ])  # (dcdS11, dcdS22)
    dsdx2 = ti.Vector([
        B[idx][0, 0] * U[0, 0] * V[0, 0] + B[idx][0, 1] * U[0, 0] * V[1, 0],
        B[idx][0, 0] * U[0, 1] * V[0, 1] + B[idx][0, 1] * U[0, 1] * V[1, 1]
    ])  #(ds11dx2, ds22dx2)
    dsdx3 = ti.Vector([
        B[idx][0, 0] * U[1, 0] * V[0, 0] + B[idx][0, 1] * U[1, 0] * V[1, 0],
        B[idx][0, 0] * U[1, 1] * V[0, 1] + B[idx][0, 1] * U[1, 1] * V[1, 1]
    ])  #(ds11dx3, ds22dx3)
    dsdx4 = ti.Vector([
        B[idx][1, 0] * U[0, 0] * V[0, 0] + B[idx][1, 1] * U[0, 0] * V[1, 0],
        B[idx][1, 0] * U[0, 1] * V[0, 1] + B[idx][1, 1] * U[0, 1] * V[1, 1]
    ])  #(ds11dx4, ds22dx4)
    dsdx5 = ti.Vector([
        B[idx][1, 0] * U[1, 0] * V[0, 0] + B[idx][1, 1] * U[1, 0] * V[1, 0],
        B[idx][1, 0] * U[1, 1] * V[0, 1] + B[idx][1, 1] * U[1, 1] * V[1, 1]
    ])  #(ds11dx5, ds22dx5)
    dsdx0 = -(dsdx2 + dsdx4)
    dsdx1 = -(dsdx3 + dsdx5)
    # constraint gradient
    dcdx2 = dcdS.dot(dsdx2)
    dcdx3 = dcdS.dot(dsdx3)
    dcdx4 = dcdS.dot(dsdx4)
    dcdx5 = dcdS.dot(dsdx5)
    dcdx0 = dcdS.dot(dsdx0)
    dcdx1 = dcdS.dot(dsdx1)
    g0 = ti.Vector([dcdx0, dcdx1])
    g1 = ti.Vector([dcdx2, dcdx3])
    g2 = ti.Vector([dcdx4, dcdx5])
    return g0, g1, g2
@ti.func
def semiEuler():
    # semi-Euler update pos & vel
    for i in range(NV):
        if (invMass[i] != 0.0):
            vel[i] += h * gravity + attractor_strength[None] * (
                attractor_pos[None] - pos[i]).normalized(1e-5)
            oldPos[i] = pos[i]
            pos[i] += h * vel[i]
@ti.func
def updteVelocity():
    # update velocity
    for i in range(NV):
        if (invMass[i] != 0.0):
            vel[i] = (pos[i] - oldPos[i]) / h
@ti.func
def solveConstraints():
    # solving constriants
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        invM0, invM1, invM2 = invMass[ia], invMass[ib], invMass[ic]
        sumInvMass = invM0 + invM1 + invM2
        if sumInvMass < 1.0e-6:
            print("wrong invMass function")
        D_s = ti.Matrix.cols([b - a, c - a])
        F[i] = D_s @ B[i]
        #print("Deformation gradient", F[i])
        U, S, V = ti.svd(F[i])
        #constraint = sqrt(((F[i] - R).transpose() @ (F[i] - R)).trace())
        # constraint violation ||F-R|| = sqrt(tr((F-R)^T(F-R))) = ||\Sigma - I ||
        constraint = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2)
        # we don't need this if condition at all
        # if constraint < 1.0e-6:
        #     break
        g0, g1, g2 = computeGradient(i, U, S, V)
        l = invM0 * g0.norm_sqr() + invM1 * g1.norm_sqr(
        ) + invM2 * g2.norm_sqr()
        # we don't need this at all, because alpha would make the Denominator not equal 0
        # if l < 1.0e-6:
        #     break
        deltaLambda = -(constraint + alpha * lagrangian[i]) / (l + alpha)
        lagrangian[i] += deltaLambda
        if (invM0 != 0.0):
            pos[ia] += invM0 * deltaLambda * g0
        if (invM1 != 0.0):
            pos[ib] += invM1 * deltaLambda * g1
        if (invM2 != 0.0):
            pos[ic] += invM2 * deltaLambda * g2

@ti.func
def computeConstriant(idx, x0, x1, x2):
    D_s = ti.Matrix.cols([x1 - x0, x2 - x0])
    F = D_s @ B[idx]
    U, S, V = ti.svd(F)
    constraint = sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2)
    return constraint


@ti.kernel
def checkGradient():
    #deformed configuration
    for i in range(NV):
        pos[i].y *= 2

    E = 0.0
    # f: x \in R^6 -> R^0 , f(x0,x1,x2) = ||F-R||
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        #  f(x)
        fx = computeConstriant(i, a, b, c)
        # ================================= finite difference gradient =======================
        delta = 1.0e-6
        dx0 = a + ti.Vector([delta, 0])
        dx1 = a + ti.Vector([0, delta])
        dx2 = b + ti.Vector([delta, 0])
        dx3 = b + ti.Vector([0, delta])
        dx4 = c + ti.Vector([delta, 0])
        dx5 = c + ti.Vector([0, delta])
        fd0 = computeConstriant(i, dx0, b, c)
        fd1 = computeConstriant(i, dx1, b, c)
        fd2 = computeConstriant(i, a, dx2, c)
        fd3 = computeConstriant(i, a, dx3, c)
        fd4 = computeConstriant(i, a, b, dx4)
        fd5 = computeConstriant(i, a, b, dx5)

        g0 = (fd0 - fx) / delta  # f(x +  delta * [1,0,0,0,0,0])- f(x) /delta
        g1 = (fd1 - fx) / delta  # f(x +  delta * [0,1,0,0,0,0])- f(x) /delta
        g2 = (fd2 - fx) / delta  # f(x +  delta * [0,0,1,0,0,0])- f(x) /delta
        g3 = (fd3 - fx) / delta  # f(x +  delta * [0,0,0,1,0,0])- f(x) /delta
        g4 = (fd4 - fx) / delta  # f(x +  delta * [0,0,0,0,1,0])- f(x) /delta
        g5 = (fd5 - fx) / delta  # f(x +  delta * [0,0,0,0,0,1])- f(x) /delta
        gg0 = ti.Vector([g0, g1
                         ])  # gradient of constraint with respect to vertex 0
        gg1 = ti.Vector([g2, g3
                         ])  # gradient of constraint with respect to vertex 1
        gg2 = ti.Vector([g4, g5
                         ])  # gradient of constraint with respect to vertex 2
        print("Finite Difference gradient: ", gg0, "---", gg1, "---", gg2)

        # =================== analytical gradient [ag0^T, ag1^T, ag2^T]^T \in R^{6x1} ===================
        D_s = ti.Matrix.cols([b - a, c - a])
        F = D_s @ B[i]
        U, S, V = ti.svd(F)
        ag0, ag1, ag2 = computeGradient(i, U, S, V)
        print("Analytical gradient:        ", ag0, "---", ag1, "---", ag2)

        # ==============================  compute gradient error ================================
        error = sqrt((ag0 - gg0).norm_sqr() + (ag1 - gg1).norm_sqr() +
                     (ag2 - gg2).norm_sqr())
        print("Error: ", error)

        E += error
    print("Sum Error: ", E)


@ti.kernel
def timestep():
    for i in range(NumSteps):
        semiEuler()
        resetLagrangian()
        for ite in range(MaxIte):
            solveConstraints()
        updteVelocity()


init_mesh()
init_pos()
pause = False
gui = ti.GUI('XPBD-FEM')
first = True
while gui.running:
    for e in gui.get_events():
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == gui.SPACE:
            pause = not pause
    mouse_pos = gui.get_cursor_pos()
    attractor_pos[None] = mouse_pos
    attractor_strength[None] = gui.is_pressed(gui.LMB) - gui.is_pressed(
        gui.RMB)
    gui.circle(mouse_pos, radius=15, color=0x336699)
    if not pause:
        #checkGradient()
        timestep()
    #     pass
    # if first:
    #     checkGradient()
    #     first = not first
    # faces = f2v.to_numpy()
    # for i in range(NF):
    #     ia, ib, ic = faces[i]
    #     a, b, c = pos[ia], pos[ib], pos[ic]
    #     gui.triangle(a, b, c, color=0x00FF00)
    # positions = pos.to_numpy()
    # gui.circles(positions, radius=2, color=0x0000FF)
    # for i in range(N + 1):
    #     k = i * (N + 1) + N
    #     staticVerts = positions[k]
    #     gui.circle(staticVerts, radius=5, color=0xFF0000)
    gui.show()
ti.kernel_profiler_print()
ti.print_profile_info()