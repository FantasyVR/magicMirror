import taichi as ti

@ti.data_oriented
class Cloth():
    def __init__(self, N, h):
        self.h = h  # timestep size
        compliance = 1.0e-2  # Fat Tissuse compliance, for more specific material,please see: http://blog.mmacklin.com/2016/10/12/xpbd-slides-and-stiffness/
        self.alpha = compliance * (1.0 / h / h)  # timestep related compliance, see XPBD paper
        self.N = N
        self.NF = 2 * N**2  # number of faces
        self.NV = (N + 1)**2  # number of vertices
        self.pos = ti.Vector.field(2, float, self.NV)
        self.oldPos = ti.Vector.field(2, float, self.NV)
        self.vel = ti.Vector.field(2, float, self.NV)  # velocity of particles
        self.invMass = ti.field(float, self.NV)  #inverse mass of particles
        self.f2v = ti.Vector.field(3, int, self.NF)  # ids of three vertices of each face
        self.B = ti.Matrix.field(2, 2, float, self.NF)  # D_m^{-1}
        self.F = ti.Matrix.field(2, 2, float, self.NF)  # deformation gradient
        self.lagrangian = ti.field(float, self.NF)  # lagrangian multipliers
        self.gravity = ti.Vector([0, -1.2])
        self.init_mesh()
        self.init_pos()
    

    @ti.kernel
    def init_pos(self):
        N, NF, f2v, B = ti.static(self.N, self.NF, self.f2v, self.B)
        pos, oldPos, vel, invMass = ti.static(self.pos, self.oldPos, self.vel, self.invMass)
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
    def init_mesh(self):
        N, f2v = ti.static(self.N, self.f2v)
        for i, j in ti.ndrange(N, N):
            k = (i * N + j) * 2
            a = i * (N + 1) + j
            b = a + 1
            c = a + N + 2
            d = a + N + 1
            f2v[k + 0] = [a, b, c]
            f2v[k + 1] = [c, d, a]


    @ti.kernel
    def resetLagrangian(self):
        for i in range(self.NF):
            self.lagrangian[i] = 0.0


    @ti.func
    def computeGradient(self,idx, U, S, V):
        B = ti.static(self.B)
        sumSigma = ti.sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2)
        if sumSigma < 0.0000001:
            sumSigma = 1.0
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
        
    @ti.kernel
    def semiEuler(self, mousePos: ti.ext_arr(), strength: ti.f32):
        # semi-Euler update pos & vel
        NV, invMass, vel, oldPos, pos, h = ti.static(self.NV, self.invMass, self.vel, self.oldPos, self.pos, self.h)
        for i in range(NV):
            if (invMass[i] != 0.0):
                attractor_pos = ti.Vector([mousePos[0], mousePos[1]])
                vel[i] += h * self.gravity + strength * (attractor_pos - pos[i]).normalized(1e-5)
                oldPos[i] = pos[i]
                pos[i] += h * vel[i]

    @ti.kernel
    def updteVelocity(self):
        NV, invMass, vel, pos, oldPos, h = ti.static(self.NV, self.invMass, self.vel, self.pos, self.oldPos, self.h)
        # update velocity
        for i in range(NV):
            if (invMass[i] != 0.0):
                vel[i] = (pos[i] - oldPos[i]) / h
    @ti.kernel
    def solveConstraints(self):
        NF, f2v, pos, invMass, B, F, lagrangian = ti.static(self.NF, self.f2v, self.pos, self.invMass, self.B, self.F, self.lagrangian)
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
            U, S, V = ti.svd(F[i])
            constraint = ti.sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2)
            g0, g1, g2 = self.computeGradient(i, U, S, V)
            l = invM0 * g0.norm_sqr() + invM1 * g1.norm_sqr(
            ) + invM2 * g2.norm_sqr()
            deltaLambda = -(constraint + self.alpha * lagrangian[i]) / (l + self.alpha)
            lagrangian[i] += deltaLambda
            if (invM0 != 0.0):
                pos[ia] += invM0 * deltaLambda * g0
            if (invM1 != 0.0):
                pos[ib] += invM1 * deltaLambda * g1
            if (invM2 != 0.0):
                pos[ic] += invM2 * deltaLambda * g2

    @ti.func
    def computeConstriant(self, idx, x0, x1, x2):
        D_s = ti.Matrix.cols([x1 - x0, x2 - x0])
        F = D_s @ self.B[idx]
        U, S, V = ti.svd(F)
        constraint = ti.sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2)
        return constraint

    def display(self, gui):
        f2v, pos, N, NF = ti.static(self.f2v, self.pos, self.N, self.NF)
        faces = f2v.to_numpy()
        for i in range(NF):
            ia, ib, ic = faces[i]
            a, b, c = pos[ia], pos[ib], pos[ic]
            gui.triangle(a, b, c, color=0x00FF00)
        positions = pos.to_numpy()
        gui.circles(positions, radius=2, color=0x0000FF)
        for i in range(N + 1):
            k = i * (N + 1) + N
            staticVerts = positions[k]
            gui.circle(staticVerts, radius=5, color=0xFF0000)