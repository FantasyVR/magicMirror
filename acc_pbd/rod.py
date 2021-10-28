import taichi as ti
import numpy as np
import pyamgx
import scipy.sparse as sparse

@ti.data_oriented
class Rod:
    def __init__(self, N, h) -> None:
        self.h = h
        self.gravity = ti.Vector([0, -9.8])
        self.N = N
        self.NC = N - 1 # number of distance constraint
        self.init_rod()

    def init_rod(self):
        N, NC, h = self.N, self.N-1, self.h
        self.pos = ti.Vector.field(2, ti.f64, N)
        self.oldPos = ti.Vector.field(2, ti.f64, N)
        self.predictionPos = ti.Vector.field(2, ti.f64, N)
        self.vel = ti.Vector.field(2, ti.f64, N)
        self.mass = ti.field(ti.f64, N)
        self.disConsIdx = ti.Vector.field(2, int, NC)  # each element store vertex indices of the constraint
        self.disConsLen = ti.field(ti.f64, NC)  # rest state (rest length of spring in this example) of each constraint
        self.gradient = ti.Vector.field(2, ti.f64, 2 * NC)  # gradient of constraints
        self.constraint = ti.field(ti.f64, NC)  # constraints violation
        self.compliance = 1.0e-6 
        self.alpha = self.compliance / h / h
        self.lagrangian = ti.field(ti.f64, NC)
        self.K = ti.Matrix.field(2, 2, ti.f64, (N, N))# geometric stiffness

    def init_amgx(self, conf_file):
        pyamgx.initialize()
        # Initialize config and resources:
        self.cfg = pyamgx.Config().create_from_file(conf_file)
        self.rsc = pyamgx.Resources().create_simple(self.cfg)

        # Create matrices and vectors:
        self.A_amgx = pyamgx.Matrix().create(self.rsc)
        self.b_amgx = pyamgx.Vector().create(self.rsc)
        self.x_amgx = pyamgx.Vector().create(self.rsc)

        # Create solver:
        self.solver_amgx = pyamgx.Solver().create(self.rsc, self.cfg)

    def detroy_amgx(self):
        self.A_amgx.destroy()
        self.x_amgx.destroy()
        self.b_amgx.destroy()
        self.solver_amgx.destroy()
        self.rsc.destroy()
        self.cfg.destroy()
        pyamgx.finalize()

    @ti.kernel
    def initRod(self):
        N, pos,oldPos, vel, mass = ti.static(self.N, self.pos, self.oldPos, self.vel, self.mass)
        for i in range(N):
            pos[i] = ti.Vector([0.5 + 0.1 * i, 0.7])
            oldPos[i] = pos[i]
            vel[i] = ti.Vector([0.0, 0.0])
            mass[i] = 1.0
        mass[0] = 0.0  # set the first particle static
        # mass[9] = 100.0


    @ti.kernel
    def initConstraint(self):
        for i in range(self.NC):
            self.disConsIdx[i] = ti.Vector([i, i + 1])
            self.disConsLen[i] = (self.pos[i + 1] - self.pos[i]).norm()


    @ti.kernel
    def semiEuler(self):
        N, pos,oldPos, vel, mass = ti.static(self.N, self.pos, self.oldPos, self.vel, self.mass)
        # semi-euler update pos & vel
        for i in range(N):
            if (mass[i] != 0.0):
                vel[i] = vel[i] + self.h * self.gravity
                oldPos[i] = pos[i]
                pos[i] = pos[i] + self.h * vel[i]
                self.predictionPos[i] = pos[i]


    @ti.kernel
    def resetLambda(self):
        for i in range(self.NC):
            self.lagrangian[i] = 0.0


    @ti.kernel
    def resetK(self):
        for i, j in ti.ndrange(self.N, self.N):
            self.K[i, j] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])


    # compute constraint vector and gradient vector
    @ti.kernel
    def computeCg(self):
        lagrangian, pos,disConsIdx, disConsLen, mass = ti.static(self.lagrangian, self.pos, self.disConsIdx, self.disConsLen, self.mass)
        for i in range(self.NC):
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
            self.constraint[i] = l - rest_len - self.alpha * lagrangian[i]
            self.gradient[2 * i + 0] = n
            self.gradient[2 * i + 1] = -n
            # geometric stiffness
            """
                k = lambda[i]/l * (I - n * n')
                K = | Hessian_{x1,x1}, Hessian_{x1,x2}   |  = | k  -k|
                    | Hessian_{x1,x2}, Hessian_{x2,x2}   |    |-k   k|
            """
            I = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
            k = lagrangian[i] / l * (I - n @ n.transpose())
            self.K[idx1, idx1] += k
            self.K[idx1, idx2] -= k
            self.K[idx2, idx1] -= k
            self.K[idx2, idx2] += k


    """
    Assemble symmetric system matrix
    A =   |  M + K      J'  |
          |  J       -alpha |
    b = |      -M(x - y) - J' * lambda      |
        | -(constraint - alpha * lambda)    |
    """
    def assemble(self, mass, p, prep, g, KK, l, c, cidx):
        N, NC = self.N, self.NC
        dim = (2 * N + NC)  # the system dimension

        A = np.zeros((dim, dim), dtype=np.float64)
        # uppper left: mass matrix
        for i in range(N):
            A[2 * i, 2 * i] = mass[i]
            A[2 * i + 1, 2 * i + 1] = mass[i]

        # uppper left: geometric stiffness
        for i in range(N):
            for j in range(N):
                A[2 * i:2 * i + 2, 2 * j:2 * j + 2] += KK[i, j]
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
            A[2 * idx1:2 * idx1 + 2, start + i] = g0
            A[2 * idx2:2 * idx2 + 2, start + i] = g1
            # xpbd lower right
            A[start + i, start + i] = -self.alpha

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
        for i in range(1, N):
            b[2 * i:2 * i + 2] = -mass[i] * (p[i] - prep[i])
        b[:2 * N] -= Gl

        self.primal_residual = np.linalg.norm(b[2:2*N])
        self.dual_residual = np.linalg.norm(b[2*N:])
        print(f">>> Primal Residual: {self.primal_residual}")
        print(f">>> Dual Residual: {self.dual_residual}")
        self.A = A[2:, 2:]
        self.b = b[2:]

    def solve(self,solver_type = 0):
        if solver_type == 0:
            return self.solve_np(self.A, self.b)
        elif solver_type == 1:
            return self.solve_amgx(self.A, self.b)

    def solve_np(self, A, b):
        return np.linalg.solve(A, b)

    def solve_amgx(self, A, b):
        #------------ Begin AMGX Solver----------------
        A_global_sparse = sparse.csr_matrix(A)
        self.A_amgx.upload_CSR(A_global_sparse)
        self.b_amgx.upload(b)
        solusion = np.zeros(b.shape, dtype=np.float64)
        self.x_amgx.upload(solusion)
        self.solver_amgx.setup(self.A_amgx)
        self.solver_amgx.solve(self.b_amgx, self.x_amgx)
        self.x_amgx.download(solusion)
        return solusion

    @ti.kernel
    def updateV(self):
        N, mass, pos, oldPos, vel,  h = ti.static(self.N, self.mass, self.pos, self.oldPos, self.vel, self.h)
        for i in range(N):
            if mass[i] != 0.0:
                vel[i] = (pos[i] - oldPos[i]) / h

    @ti.kernel
    def updatePosLambda(self, dx: ti.ext_arr()):
        N, NC = self.N, self.NC
        for i in range(N - 1):
            self.pos[i + 1] += ti.Vector([dx[2 * i + 0], dx[2 * i + 1]])
        for i in range(NC):
            self.lagrangian[i] += dx[2*(N-1)+i]

    @ti.kernel 
    def applyChebyshev(self, omega: ti.f32):
        for i in range(self.N-1):
            self.pos[i+1] *= omega
            self.pos[i+1] += (1-omega) * self.oldPos[i+1]

    def display(self, gui):
        position = self.pos.to_numpy()
        begin = position[:-1]
        end = position[1:]
        gui.lines(begin, end, radius=3, color=0x0000FF)
        gui.circles(self.pos.to_numpy(), radius=5, color=0xffaa33)