import taichi as ti

ti.init(arch=ti.cuda)
N = 50
NV = (N + 1)**2
NT = 2 * N**2
NE = 2 * N * (N + 1) + N**2
pos = ti.Vector.field(3, ti.f32, shape=NV)
tri = ti.field(ti.i32, shape=3 * NT)
edge = ti.Vector.field(2, ti.i32, shape=NE)

old_pos = ti.Vector.field(3, ti.f32, NV)
pre_pos = ti.Vector.field(3, ti.f32, NV)
next_pos = ti.Vector.field(3, ti.f32, NV)
inv_mass = ti.field(ti.f32, NV)
vel = ti.Vector.field(3, ti.f32, NV)
rest_len = ti.field(ti.f32, NE)
h = 0.05
MaxIte = 200

paused = ti.field(ti.i32, shape=())


@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, N + 1):
        idx = i * (N + 1) + j
        pos[idx] = ti.Vector([i / N, 0.5, j / N])
        inv_mass[idx] = 1.0
    inv_mass[N] = 0.0
    inv_mass[NV-1] = 0.0


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
def semi_euler():
    gravity = ti.Vector([0.0, -0.1, 0.0])
    for i in range(NV):
        if inv_mass[i] != 0.0:
            vel[i] += h * gravity
            old_pos[i] = pos[i]
            pos[i] += h * vel[i]

@ti.kernel
def solve_constraints() -> ti.f32:
    dual_residual = 0.0
    for i in range(NE):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        dis = next_pos[idx0] - next_pos[idx1]
        constraint = dis.norm() - rest_len[i]
        dual_residual += constraint**2
        gradient = dis.normalized()
        l = -constraint / (invM0 + invM1)
        if invM0 != 0.0:
            next_pos[idx0] += 0.5 * invM0 * l * gradient
        if invM1 != 0.0:
            next_pos[idx1] -= 0.5 * invM1 * l * gradient
    return dual_residual

@ti.kernel
def update_vel():
    for i in range(NV):
        if inv_mass[i] != 0.0:
            vel[i] = (pos[i] - old_pos[i]) / h

@ti.kernel 
def copy_field(source: ti.template(), des: ti.template()):
    for x in source:
        des[x] = source[x]

@ti.kernel 
def apply_chebyshev(omega: ti.f32):
    for i in range(NV):
        next_pos[i] = omega * ( next_pos[i] - pre_pos[i]) + pre_pos[i]

def step():
    semi_euler()
    omega, rho = 1.0, 0.8
    use_chebyshev = True
    f = open("data/pbd_use_chebyshev.txt", 'a')
    for i in range(MaxIte):
        copy_field(pos, next_pos)
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
        copy_field(pos, pre_pos)
        copy_field(next_pos, pos)
    f.close()
    update_vel()


init_pos()
init_tri()
init_edge()

window = ti.ui.Window("Display Mesh", (1024, 1024))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = camera = ti.ui.Camera()
camera.position(0.5, 0.0, 2.5)
camera.lookat(0.5, 0.5, 0.0)
camera.fov(90)

paused[None] = 0
while window.running:
    for e in window.get_events(ti.ui.PRESS):
        if e.key in [ti.ui.ESCAPE]:
            exit()
    if window.is_pressed(ti.ui.SPACE):
        paused[None] = not paused[None]

    if not paused[None]:
        step()

    camera.track_user_inputs(window, movement_speed=0.003, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

    scene.mesh(pos, tri, color=(1.0,1.0,1.0), two_sided=True)
    scene.particles(pos, radius=0.01, color=(0.6,0.0,0.0))
    canvas.scene(scene)
    window.show()
