import taichi as ti
import numpy as np 

ti.init(arch=ti.cuda)
N = 20
NV = (N + 1)**2
NT = 2 * N**2
NE = 2 * N * (N + 1) + N**2
pos = ti.Vector.field(3, ti.f32, shape=NV)
tri = ti.field(ti.i32, shape=3 * NT)
edge = ti.Vector.field(2, ti.i32, shape=NE)

old_pos = ti.Vector.field(3, ti.f32, NV)
inv_mass = ti.field(ti.f32, NV)
vel = ti.Vector.field(3, ti.f32, NV)
rest_len = ti.field(ti.f32, NE)
h = 0.01
MaxIte = 100

paused = ti.field(ti.i32, shape=())

gradient = ti.Vector.field(3, ti.f32, NE)
constraint = ti.field(ti.f32, NE)

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
        pos[i] += ti.Vector([random_pos[i, 0], random_pos[i, 1],  random_pos[i, 2]])


def init_random_pos():
    f = open("random.txt", 'r')
    random_pos = [float(l) for l in f.readlines()]
    assert len(random_pos) // 3 >= NV
    random_pos = np.asarray(random_pos,
                            dtype=np.float32).reshape(len(random_pos) // 3, 3)
    move_random(random_pos)


@ti.kernel
def semi_euler():
    for i in range(NV):
        old_pos[i] = pos[i]
        pos[i] += h * vel[i]

@ti.func 
def compute_gradient_constraint():
    for i in range(NE):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        constraint[i] = dis.norm() - rest_len[i]
        gradient[i] = dis.normalized()

@ti.kernel
def solve_constraints():
    compute_gradient_constraint()
    for i in range(NE):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        l = -constraint[i] / (invM0 + invM1)
        pos[idx0] += 0.5 * invM0 * l * gradient[i]
        pos[idx1] -= 0.5 * invM1 * l * gradient[i]

@ti.kernel
def update_vel():
    for i in range(NV):
        vel[i] = (pos[i] - old_pos[i]) / h

def step():
    semi_euler()
    for i in range(MaxIte):
        solve_constraints()
    update_vel()


init_pos()
init_tri()
init_edge()

init_random_pos()

window = ti.ui.Window("Display Mesh", (1024, 1024))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = camera = ti.ui.Camera()
camera.position(0.5, 0.0, 2.5)
camera.lookat(0.5, 0.5, 0.0)
camera.fov(90)

paused[None] = 1
while window.running:
    for e in window.get_events(ti.ui.PRESS):
        if e.key in [ti.ui.ESCAPE]:
            exit()
    if window.is_pressed(ti.ui.SPACE):
        paused[None] = not paused[None]

    if not paused[None]:
        step()
        paused[None] = not paused[None]

    camera.track_user_inputs(window, movement_speed=0.003, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

    scene.mesh(pos, tri, color=(1.0,1.0,1.0), two_sided=True)
    scene.particles(pos, radius=0.01, color=(0.6,0.0,0.0))
    canvas.scene(scene)
    window.show()
