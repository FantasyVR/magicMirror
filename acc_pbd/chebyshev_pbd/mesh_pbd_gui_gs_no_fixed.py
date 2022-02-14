import taichi as ti
import numpy as np 

# ti.init(arch=ti.cpu, cpu_max_num_threads=1)
ti.init(arch=ti.cpu)


N = 50
NV = (N+1)**2
NE = (N+1) * N * 2 + N**2

positions = ti.Vector.field(2, ti.f32, NV)
old_positions = ti.Vector.field(2, ti.f32, NV)
next_positions = ti.Vector.field(2, ti.f32, NV)
pre_positions = ti.Vector.field(2, ti.f32, NV)

edge_indices = ti.Vector.field(2, ti.i32, NE)

inv_mass =ti.field(ti.f32, NV)
velocities = ti.Vector.field(2, ti.f32, NV)

rest_len = ti.field(ti.f32, NE)

@ti.kernel 
def init_pos():
    step = 1/N * 0.5
    for i, j in ti.ndrange(N+1, N+1):
        positions[i * (N+1) + j]  = ti.Vector([i, j]) * step  + ti.Vector([0.25,0.25])
    for i in range(NV):
        old_positions[i] = positions[i]
        inv_mass[i] = 1.0

@ti.kernel 
def init_edge():
    for i, j in ti.ndrange(N+1, N):
        a = i * (N + 1) + j
        edge_indices[i * N + j] = ti.Vector([a, a+1])
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
        idx0, idx1  = edge_indices[i] 
        rest_len[i] = (positions[idx0] - positions[idx1]).norm()

@ti.kernel
def move_random(random_pos: ti.ext_arr()):
    for i in range(NV):
        positions[i] += ti.Vector([random_pos[i, 0], random_pos[i, 1]])

def init_random_pos():
    f = open("random.txt", 'r')
    random_pos = [float(l) for l in f.readlines()]
    assert len(random_pos)//2 >= NV
    random_pos = np.asarray(random_pos, dtype=np.float32).reshape(len(random_pos)//2, 2)
    move_random(random_pos)

@ti.kernel 
def semi_euler(h: ti.f32):
    gravity = ti.Vector([0.0, -9.8])
    for i in range(NV):
        if inv_mass[i] == 0.0:
            continue
        # velocities[i] += h * gravity
        old_positions[i] = positions[i]
        positions[i] += h * velocities[i]

@ti.kernel 
def copy_field(source: ti.template(), des: ti.template()):
    for x in source:
        des[x] = source[x]

@ti.kernel 
def solve_constraints() -> ti.f32:
    dual_residual = 0.0
    for i in range(NE):
        idx0, idx1  = edge_indices[i] 
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        dis = next_positions[idx0] - next_positions[idx1]
        constraint = dis.norm() - rest_len[i]
        dual_residual += constraint**2
        gradient = dis.normalized()
        l = -constraint / (invM0 + invM1)
        if invM0 != 0.0:
            next_positions[idx0] += invM0 * l * gradient
        if invM1 != 0.0:
            next_positions[idx1] -= invM1 * l * gradient
    return dual_residual
@ti.kernel
def update_v(h: ti.f32):
    for i in range(NV):
        if inv_mass[i] == 0.0:
            continue
        velocities[i] = (positions[i] - old_positions[i])/h
@ti.kernel 
def apply_chebyshev(omega: ti.f32):
    for i in range(NV):
        next_positions[i] = omega * ( next_positions[i] - pre_positions[i]) + pre_positions[i]


def update(h, maxIte):
    semi_euler(h)
    omega, rho = 1.0, 0.8
    use_chebyshev = True
    f = open("data/pbd_use_chebyshev.txt", 'a')
    for i in range(maxIte):
        copy_field(positions, next_positions)
        dual_residual = solve_constraints()
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

init_pos()
init_edge()
init_rest_len()

init_random_pos()
gui = ti.GUI("Diplay tri mesh", res=(600,600))
pause = False
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
    gui.lines(np.asarray(begin_line), np.asarray(end_line), radius=2, color=0x0000FF) 
    gui.circles(positions.to_numpy(), radius=6, color=0xffaa33)
    gui.show()