import taichi as ti
import numpy as np
from numpy.linalg import inv

@ti.data_oriented
class Cloth():
    def __init__(self, N, h):
        self.h = h  # timestep size
        compliance = 1.0e-5  # Fat Tissuse compliance, for more specific material,please see: http://blog.mmacklin.com/2016/10/12/xpbd-slides-and-stiffness/
        self.alpha = compliance * (1.0 / h / h)  # timestep related compliance, see XPBD paper
        self.N = N
        self.NF = 2 * N**2  # number of faces
        self.NV = (N + 1)**2  # number of vertices
        self.pos = ti.Vector.field(2, float, self.NV)
        self.oldPos = ti.Vector.field(2, float, self.NV)
        self.prePos = ti.Vector.field(2, float, self.NV)
        self.vel = ti.Vector.field(2, float, self.NV)  # velocity of particles
        self.invMass = ti.field(float, self.NV)  #inverse mass of particles
        self.f2v = ti.Vector.field(3, int, self.NF)  # ids of three vertices of each face
        self.B = ti.Matrix.field(2, 2, float, self.NF)  # D_m^{-1}
        self.F = ti.Matrix.field(2, 2, float, self.NF)  # deformation gradient
        self.lagrangian = ti.field(float, self.NF)  # lagrangian multipliers
        self.old_lagrangian = ti.field(float, self.NF)  # lagrangian multipliers
        self.gravity = ti.Vector([0, -1.2])

        self.gradient = ti.Vector.field(2,float, 3 * self.NF)
        self.dLambda = ti.field(float, self.NF)
        self.c = ti.field(float, self.NF)
        self.relax_ratio = 0.5
        self.init_mesh()
        self.init_pos()
    

    @ti.kernel
    def init_pos(self):
        N, NF, f2v, B = ti.static(self.N, self.NF, self.f2v, self.B)
        pos, oldPos, vel, invMass = ti.static(self.pos, self.oldPos, self.vel, self.invMass)
        for i, j in ti.ndrange(N + 1, N + 1):
            k = i * (N + 1) + j
            pos[k] = ti.Vector([i, j]) / N * 0.5 + ti.Vector([0.25, 0.1])
            self.prePos[k] = pos[k]
            oldPos[k] = pos[k]
            vel[k] = ti.Vector([0, 0])
            invMass[k] = 1.0
        # for i in range(N + 1):
        #     k = i * (N + 1) + N
        #     invMass[k] = 0.0
        # k0 = N
        # k1 = (N + 2) * N
        # invMass[k0] = 0.0
        # invMass[k1] = 0.0
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
        isSuccess = True 
        sumSigma = ti.sqrt((S[0, 0] - 1)**2 + (S[1, 1] - 1)**2)
        if sumSigma < 0.0000001:
            sumSigma = 1.0
            isSuccess = False 
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
        return g0, g1, g2, isSuccess
        
    @ti.kernel
    def semiEuler(self, mousePos: ti.ext_arr(), strength: ti.f32):
        # semi-Euler update pos & vel
        NV, invMass, vel, oldPos, pos, h = ti.static(self.NV, self.invMass, self.vel, self.oldPos, self.pos, self.h)
        for i in range(NV):
            if (invMass[i] != 0.0):
                attractor_pos = ti.Vector([mousePos[0], mousePos[1]])
                vel[i] += h * self.gravity + strength * (attractor_pos - pos[i]).normalized(1e-5)
                self.prePos[i] = oldPos[i]
                oldPos[i] = pos[i]
                pos[i] += h * vel[i]


    @ti.kernel
    def computeGradientVector(self) -> ti.f32:
        NF, f2v, pos, invMass, B, F, lagrangian = ti.static(self.NF, self.f2v, self.pos, self.invMass, self.B, self.F, self.lagrangian)
        epsilon = 0.0
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
            g0, g1, g2, isSuccess = self.computeGradient(i, U, S, V)
            if isSuccess:
                l = invM0 * g0.norm_sqr() + invM1 * g1.norm_sqr(
                ) + invM2 * g2.norm_sqr()
                residual =  constraint + self.alpha * lagrangian[i]
                self.c[i] = residual
                epsilon += residual**2
                self.dLambda[i] = - residual / (l + self.alpha)
                lagrangian[i] += self.dLambda[i]
                self.gradient[3 * i + 0]  = g0
                self.gradient[3 * i + 1]  = g1
                self.gradient[3 * i + 2]  = g2
        return epsilon

    """
    Assemble system matrix
    A =   |  M      -J'  |
          |  J   alpha |
    b = |u| = |             0                   |
        |v|   | -(constraint + alpha * lambda)  |
    """
    def solve_global(self):
        NV, NF = self.NV, self.NF
        invmass = self.invMass
        dim = 2 * NV  # the system dimension
        g = self.gradient.to_numpy()
        inv_M = np.zeros((dim, dim), dtype=np.float32)
        # uppper left: mass matrix
        for i in range(NV):
            inv_M[2 * i, 2 * i] = invmass[i]
            inv_M[2 * i + 1, 2 * i + 1] = invmass[i]

        # gradient matrix
        G = np.zeros((2 * NV, NF))
        for i in range(NF):
            ia, ib, ic = self.f2v[i]
            g0 = g[3 * i + 0]
            g1 = g[3 * i + 1]
            g2 = g[3 * i + 2]
            G[2 * ia:2 * ia + 2, i] = g0
            G[2 * ib:2 * ib + 2, i] = g1
            G[2 * ic:2 * ic + 2, i] = g2

        # compliance matrix
        complianceMatrix = np.zeros((NF, NF), dtype=np.float32)
        np.fill_diagonal(complianceMatrix, self.alpha)

        # RHS
        v = -self.c.to_numpy()

        S = complianceMatrix + np.transpose(G) @ inv_M @ G # schur complement matrix

        # Jacobi solver
        Diag = np.array(np.diag(S))
        LU = S
        for i in range(self.NF):
            S[i,i] = 0.0
        dl = np.zeros(NF,dtype=np.float32)
        for jacIte in range(1):
            for i in range(NF):
                sum = 0.0
                for j in range(NF):
                    if i != j:
                        sum += LU[i, j] * dl[j]
                dl[i] = (v[i] - sum)/Diag[i]
        # Update delta X
        dx = G @ dl
        return dx, dl
    
    @ti.kernel
    def updatePosLambda(self, dx: ti.ext_arr(), dl: ti.ext_arr()):
        for i in range(self.NV):
            self.pos[i] += ti.Vector([dx[2 * i + 0], dx[2 * i + 1]])
            if self.pos[i][1] < 0.0:
                self.pos[i][1] = 0.0
        for i in range(self.NF):
            self.old_lagrangian[i] = self.lagrangian[i]
            self.lagrangian[i] += dl[i]


    @ti.kernel
    def applyPrimalChebyshev(self, omega: ti.f32):
        for i in range(self.NV):
            self.pos[i] *= omega
            self.pos[i] += (1-omega) * self.oldPos[i]
    
    @ti.kernel
    def applyDualChebyshev(self, omega: ti.f32):
        for i in range(self.NV-1):
            self.lagrangian[i] *= omega
            self.lagrangian[i] += (1-omega) * self.old_lagrangian[i]

    @ti.kernel
    def updteVelocity(self):
        NV, invMass, vel, pos, oldPos, h = ti.static(self.NV, self.invMass, self.vel, self.pos, self.oldPos, self.h)
        # update velocity
        for i in range(NV):
            if (invMass[i] != 0.0):
                vel[i] = (pos[i] - oldPos[i]) / h

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
