import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, cpu_max_num_threads=1, default_fp=ti.f64)
# ti.init(arch=ti.gpu)

N = 6
NV = (N + 1)**2
NE = (N + 1) * N * 2 + N**2

positions = ti.Vector.field(2, ti.f64, NV)
old_positions = ti.Vector.field(2, ti.f64, NV)
next_positions = ti.Vector.field(2, ti.f64, NV)
pre_positions = ti.Vector.field(2, ti.f64, NV)

edge_indices = ti.Vector.field(2, ti.i32, NE)

inv_mass = ti.field(ti.f64, NV)
velocities = ti.Vector.field(2, ti.f64, NV)

rest_len = ti.field(ti.f64, NE)

gradient = ti.Vector.field(2, ti.f64, NE)
constraint = ti.field(ti.f64, NE)


@ti.kernel
def init_pos():
    step = 1 / N * 0.5
    for i, j in ti.ndrange(N + 1, N + 1):
        positions[i * (N + 1) +
                  j] = ti.Vector([i, j]) * step + ti.Vector([0.25, 0.25])
    for i in range(NV):
        old_positions[i] = positions[i]
        inv_mass[i] = 1.0


@ti.kernel
def init_edge():
    for i, j in ti.ndrange(N + 1, N):
        a = i * (N + 1) + j
        edge_indices[i * N + j] = ti.Vector([a, a + 1])
    start = N * (N + 1)
    for i, j in ti.ndrange(N, N + 1):
        a = i * (N + 1) + j
        edge_indices[start + i + j * N] = ti.Vector([a, a + N + 1])
    start = 2 * N * (N + 1)
    for i, j in ti.ndrange(N, N):
        edge_idx = start + i * N + j
        pos_idx = i * (N + 1) + j
        if (i + j) % 2 == 0:
            edge_indices[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
        else:
            edge_indices[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])


@ti.kernel
def init_rest_len():
    for i in range(NE):
        idx0, idx1 = edge_indices[i]
        rest_len[i] = (positions[idx0] - positions[idx1]).norm()


@ti.kernel
def move_random(random_pos: ti.ext_arr()):
    for i in range(NV):
        positions[i] += ti.Vector([random_pos[i, 0], random_pos[i, 1]])


def init_random_pos():
    f = open("random.txt", 'r')
    random_pos = [float(l) for l in f.readlines()]
    assert len(random_pos) // 2 >= NV
    random_pos = np.asarray(random_pos,
                            dtype=np.float64).reshape(len(random_pos) // 2, 2)
    move_random(random_pos)


@ti.kernel
def semi_euler(h: ti.f64):
    for i in range(NV):
        old_positions[i] = positions[i]
        positions[i] += h * velocities[i]


@ti.kernel
def copy_field(source: ti.template(), des: ti.template()):
    for x in source:
        des[x] = source[x]


@ti.kernel
def compute_gradient_constraint() -> ti.f64:
    dual_residual = 0.0
    for i in range(NE):
        idx0, idx1 = edge_indices[i]
        dis = next_positions[idx0] - next_positions[idx1]
        constraint[i] = dis.norm() - rest_len[i]
        gradient[i] = dis.normalized()
        dual_residual += constraint[i]**2
    return dual_residual


@ti.kernel
def correct(delta_x: ti.ext_arr()):
    for i in range(NV):
        next_positions[i] += ti.Vector(
            [delta_x[2 * i + 0], delta_x[2 * i + 1]])


def solve_constraints(ite):
    g = gradient.to_numpy()
    G = np.zeros((2 * NV, NE), np.float64)
    e_indices = edge_indices.to_numpy()
    for i in range(NE):
        idx0, idx1 = e_indices[i]
        G[2 * idx0:2 * idx0 + 2, i] = g[i, :]
        G[2 * idx1:2 * idx1 + 2, i] = -g[i, :]
    A = np.transpose(G) @ G
    b = constraint.to_numpy()
    l = np.zeros(NE, np.float64)

    # Direct Solver
    # l = np.linalg.solve(A, -b)

    # Jacobi
    # for i in range(NE):
    #     l[i] = -b[i] / A[i, i]

    # Gauss-Seidel
    for i in range(NE):
        sum = 0.0
        for j in range(NE):
            if i != j:
                sum += A[i, j] * l[j]
        l[i]  = -(b[i]-sum)/A[i,i]

    delta_x =  0.2 * G @ l
    correct(delta_x)


def solve(ite):
    dual_residual = compute_gradient_constraint()
    solve_constraints(ite)
    return dual_residual


@ti.kernel
def update_v(h: ti.f64):
    for i in range(NV):
        velocities[i] = (positions[i] - old_positions[i]) / h


@ti.kernel
def apply_chebyshev(omega: ti.f64):
    for i in range(NV):
        next_positions[i] = omega * (next_positions[i] -
                                     pre_positions[i]) + pre_positions[i]


def update(h, maxIte):
    semi_euler(h)
    omega, rho = 1.0, 0.8
    use_chebyshev = False
    f = open("data/pbd_no_chebyshev.txt", 'a')
    for i in range(maxIte):
        print(f"================ Iteration: {i} =============")
        copy_field(positions, next_positions)
        dual_residual = solve(i)
        f.write(f"{dual_residual} \n")
        if use_chebyshev:
            if i < 10:
                omega = 1.0
            elif i == 10:
                omega = 2 / (2 - rho * rho)
            else:
                omega = 4 / (4 - rho * rho * omega)
            apply_chebyshev(omega)
        copy_field(positions, pre_positions)
        copy_field(next_positions, positions)
    update_v(h)


def write_pos():
    np.set_printoptions(linewidth=np.inf, precision=3, suppress=True)
    f = open("pos", "a+")
    f.write("positions: \n")
    pos = positions.to_numpy()
    for i, p in enumerate(pos):
        f.write(f"{i}: {p}\n")

    f.write("indices: \n")
    indices = edge_indices.to_numpy()
    for i, p in enumerate(indices):
        f.write(f"{i}: {p} \n")
    f.close()

    n = pos[0] - pos[1]
    print(f"normal: {n/np.linalg.norm(n)}")


init_pos()
init_edge()
init_rest_len()

init_random_pos()
# write_pos()

gui = ti.GUI("Diplay tri mesh", res=(600, 600))
pause = True
h = 0.01
maxIte = 100
while gui.running:
    gui.get_event(ti.GUI.PRESS)
    if gui.is_pressed(ti.GUI.ESCAPE):
        gui.running = False
    elif gui.is_pressed(ti.GUI.SPACE):
        pause = not pause

    if not pause:
        update(h, maxIte)

    poses = positions.to_numpy()
    edges = edge_indices.to_numpy()
    begin_line, end_line = [], []
    for i in range(edges.shape[0]):
        idx0, idx1 = edges[i]
        begin_line.append(poses[idx0])
        end_line.append(poses[idx1])
    gui.lines(np.asarray(begin_line),
              np.asarray(end_line),
              radius=2,
              color=0x0000FF)
    gui.circles(positions.to_numpy(), radius=6, color=0xffaa33)
    gui.show()
