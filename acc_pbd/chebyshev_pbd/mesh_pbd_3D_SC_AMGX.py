from email.policy import default
import taichi as ti
import pyamgx
import numpy as np
import scipy.sparse as sparse

pyamgx.initialize()

# Initialize config and resources:
cfg = pyamgx.Config().create_from_dict({
    "config_version": 2,
    "solver": {
        "print_grid_stats": 0,
        "solver": "AMG",
        "algorithm": "AGGREGATION",
        "selector": "SIZE_4",
        "print_solve_stats": 0,
        "smoother": "JACOBI_L1",
        "presweeps": 0,
        "postsweeps": 3,
        "obtain_timings": 0,
        "max_iters": 200,
        "monitor_residual": 1,
        "convergence": "ABSOLUTE",
        "scope": "main",
        "max_levels": 100,
        "cycle": "CG",
        "tolerance": 1e-06,
        "norm": "L2"
    }
})

rsc = pyamgx.Resources().create_simple(cfg)

# Create matrices and vectors:
A_glb = pyamgx.Matrix().create(rsc)
b_glb = pyamgx.Vector().create(rsc)
x_glb = pyamgx.Vector().create(rsc)

# Create solver:
solver = pyamgx.Solver().create(rsc, cfg)

ti.init(arch=ti.cpu, default_fp=ti.f64)
N = 10
NV = (N + 1)**2
NT = 2 * N**2
NE = 2 * N * (N + 1) + N**2
pos = ti.Vector.field(3, ti.f64, shape=NV)
tri = ti.field(ti.i32, shape=3 * NT)
edge = ti.Vector.field(2, ti.i32, shape=NE)

old_pos = ti.Vector.field(3, ti.f64, NV)
pre_pos = ti.Vector.field(3, ti.f64, NV)
next_pos = ti.Vector.field(3, ti.f64, NV)
inv_mass = ti.field(ti.f64, NV)
vel = ti.Vector.field(3, ti.f64, NV)
rest_len = ti.field(ti.f64, NE)
h = 0.01
MaxIte = 100

paused = ti.field(ti.i32, shape=())

gradient = ti.Vector.field(3, ti.f64, NE)
constraint = ti.field(ti.f64, NE)


@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, N + 1):
        idx = i * (N + 1) + j
        pos[idx] = ti.Vector([i / N, 0.5, j / N])
        inv_mass[idx] = 1.0


@ti.kernel
def init_tri():
    for i, j in ti.ndrange(N, N):
        tri_idx = 6 * (i * N + j)
        pos_idx = i * (N + 1) + j
        if (i + j) % 2 == 0:
            tri[tri_idx + 0] = pos_idx
            tri[tri_idx + 1] = pos_idx + N + 2
            tri[tri_idx + 2] = pos_idx + 1
            tri[tri_idx + 3] = pos_idx
            tri[tri_idx + 4] = pos_idx + N + 1
            tri[tri_idx + 5] = pos_idx + N + 2
        else:
            tri[tri_idx + 0] = pos_idx
            tri[tri_idx + 1] = pos_idx + N + 1
            tri[tri_idx + 2] = pos_idx + 1
            tri[tri_idx + 3] = pos_idx + 1
            tri[tri_idx + 4] = pos_idx + N + 1
            tri[tri_idx + 5] = pos_idx + N + 2


@ti.kernel
def init_edge():
    for i, j in ti.ndrange(N + 1, N):
        edge_idx = i * N + j
        pos_idx = i * (N + 1) + j
        edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
    start = N * (N + 1)
    for i, j in ti.ndrange(N, N + 1):
        edge_idx = start + j * N + i
        pos_idx = i * (N + 1) + j
        edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
    start = 2 * N * (N + 1)
    for i, j in ti.ndrange(N, N):
        edge_idx = start + i * N + j
        pos_idx = i * (N + 1) + j
        if (i + j) % 2 == 0:
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
        else:
            edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])
    for i in range(NE):
        idx1, idx2 = edge[i]
        p1, p2 = pos[idx1], pos[idx2]
        rest_len[i] = (p1 - p2).norm()


@ti.kernel
def move_random(random_pos: ti.ext_arr()):
    for i in range(NV):
        pos[i] += ti.Vector(
            [random_pos[i, 0], random_pos[i, 1], random_pos[i, 2]])


def init_random_pos():
    f = open("random.txt", 'r')
    random_pos = [float(l) for l in f.readlines()]
    assert len(random_pos) // 3 >= NV
    random_pos = np.asarray(random_pos,
                            dtype=np.float64).reshape(len(random_pos) // 3, 3)
    move_random(random_pos)


@ti.kernel
def semi_euler():
    for i in range(NV):
        old_pos[i] = pos[i]
        pos[i] += h * vel[i]


@ti.kernel
def compute_gradient_constraint() -> ti.f64:
    dual_residual = 0.0
    for i in range(NE):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        constraint[i] = dis.norm() - rest_len[i]
        gradient[i] = dis.normalized()
        dual_residual += constraint[i]**2
    return dual_residual


@ti.kernel
def correct(delta_x: ti.ext_arr()):
    for i in range(NV):
        next_pos[i] += ti.Vector(
            [delta_x[3 * i + 0], delta_x[3 * i + 1], delta_x[3 * i + 2]])


def direct_solve(A, b):
    # print("direct solve")
    l = np.zeros(NE, np.float64)
    l = np.linalg.solve(A, b)
    return l


def GS_solve(A, b):
    # print("GS_solve")
    l = np.zeros(NE, np.float64)
    for i in range(NE):
        l[i] = b[i] / A[i, i]
    for k in range(NE):
        for i in range(NE):
            if i != k:
                l[i] -= l[k] * A[i, k]
    return l


def Jacobi_solve(A, b):
    # print("Jacobi_solve")
    l = np.zeros(NE, np.float64)
    for i in range(NE):
        l[i] = b[i] / A[i, i]
    return l


def AMGX_solve(A, b):
    # print("AMGX solve")
    M = sparse.csr_matrix(A)
    rhs = b
    sol = np.zeros(NE, np.float64)
    A_glb.upload_CSR(M)
    b_glb.upload(rhs)
    x_glb.upload(sol)

    # Setup and solve system:
    solver.setup(A_glb)
    solver.solve(b_glb, x_glb)

    # Download solution
    x_glb.download(sol)
    return sol


def solve_constraints(solve_id):
    g = gradient.to_numpy()
    G = np.zeros((3 * NV, NE), np.float64)
    e_indices = edge.to_numpy()
    for i in range(NE):
        idx0, idx1 = e_indices[i]
        G[3 * idx0:3 * idx0 + 3, i] = g[i]
        G[3 * idx1:3 * idx1 + 3, i] = -g[i]
    A = np.transpose(G) @ G
    b = -constraint.to_numpy()
    fuc = [AMGX_solve, Jacobi_solve, GS_solve, direct_solve]
    l = fuc[solve_id](A, b)
    delta_x = 0.2 * G @ l
    correct(delta_x)


def solve(solve_id):
    dual_residual = compute_gradient_constraint()
    solve_constraints(solve_id)
    return dual_residual


@ti.kernel
def update_vel():
    for i in range(NV):
        vel[i] = (pos[i] - old_pos[i]) / h


@ti.kernel
def copy_field(source: ti.template(), des: ti.template()):
    for x in source:
        des[x] = source[x]


@ti.kernel
def apply_chebyshev(omega: ti.f64):
    for i in range(NV):
        next_pos[i] = omega * (next_pos[i] - pre_pos[i]) + pre_pos[i]


def step(i):
    print(f"frame: {i}")
    semi_euler()
    omega, rho = 1.0, 0.8
    use_chebyshev = False
    solver_id = 3
    output_file = ["pbd_AMGX", "pbd_Jacobi", "pbd_GS", "pbd_Direct"]
    f = open(f"data/{output_file[solver_id]}.txt", 'a')
    for i in range(MaxIte):
        copy_field(pos, next_pos)
        # 0: AMGX 1:Jacobi 2:GS 3: Direct
        dual_residual = solve(solver_id)
        f.write(f"{dual_residual}  \n")
        if use_chebyshev:
            if i < 10:
                omega = 1.0
            elif i == 10:
                omega = 2 / (2 - rho * rho)
            else:
                omega = 4 / (4 - rho * rho * omega)
            apply_chebyshev(omega)
        copy_field(pos, pre_pos)
        copy_field(next_pos, pos)
    f.close()
    update_vel()


def clean_up():
    A_glb.destroy()
    x_glb.destroy()
    b_glb.destroy()
    solver.destroy()
    rsc.destroy()
    cfg.destroy()
    pyamgx.finalize()


init_pos()
init_tri()
init_edge()

init_random_pos()

for i in range(10):
    step(i)

clean_up()

# window = ti.ui.Window("Display Mesh", (1024, 1024))
# canvas = window.get_canvas()
# scene = ti.ui.Scene()
# camera = ti.ui.make_camera()
# camera.position(0.5, 0.0, 2.5)
# camera.lookat(0.5, 0.5, 0.0)
# camera.fov(90)

# paused[None] = 0
# while window.running:
#     for e in window.get_events(ti.ui.PRESS):
#         if e.key in [ti.ui.ESCAPE]:
#             exit()
#     if window.is_pressed(ti.ui.SPACE):
#         paused[None] = not paused[None]

#     if not paused[None]:
#         step()

#     camera.track_user_inputs(window, movement_speed=0.003, hold_key=ti.ui.RMB)
#     scene.set_camera(camera)
#     scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

#     scene.mesh(pos, tri, color=(1.0, 1.0, 1.0), two_sided=True)
#     scene.particles(pos, radius=0.01, color=(0.6, 0.0, 0.0))
#     canvas.scene(scene)
#     window.show()
