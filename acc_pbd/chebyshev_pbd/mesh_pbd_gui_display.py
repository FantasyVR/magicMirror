import taichi as ti

ti.init(arch=ti.cuda)
N = 10
NV = (N + 1)**2
NT = 2 * N**2
NE = N * (N + 1) + N**2
pos = ti.Vector.field(3, ti.f32, shape=NV)
per_color = ti.Vector.field(3, ti.f32, shape=NV)
tri = ti.field(ti.i32, shape=3 * NT)
edg = ti.Vector.field(2, ti.i32, shape=NE)

paused = ti.field(ti.i32, shape=())


@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, N + 1):
        idx = i * (N + 1) + j
        pos[idx] = ti.Vector([i / N, j / N, 0.0])
        per_color[idx] = ti.Vector([idx / NV, idx / NV, 1.0])


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
        edg_idx = i * N + j
        pos_idx = i * (N + 1) + j
        edg[edg_idx] = ti.Vector([pos_idx, pos_idx + 1])
    start = N * (N + 1)
    for i, j in ti.ndrange(N, N + 1):
        edg_idx = start + j * N + i
        pos_idx = i * (N + 1) + j
        edg[edg_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
    start = 2 * N * (N + 1)
    for i, j in ti.ndrange(N, N):
        edg_idx = start + i * N + j
        pos_idx = i * (N + 1) + j
        if (i+j) % 2 == 0:
            edg[edg_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
        else:
            edg[edg_idx] = ti.Vector([pos_idx+1, pos_idx + N + 1])


init_pos()
init_tri()
init_edge()

window = ti.ui.Window("Display Mesh", (1024, 1024))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(0.5, 0.5, 2.5)
camera.lookat(0.5, 0.5, 0.0)
camera.fov(55)

while window.running:
    for e in window.get_events(ti.ui.PRESS):
        if e.key in [ti.ui.ESCAPE]:
            exit()
        elif e.key == ti.ui.SPACE:
            paused[None] = not paused[None]

    camera.track_user_inputs(window, movement_speed=0.003, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

    scene.mesh(pos, tri, color=(0.5, 0.5, 0.5), two_sided=True)
    scene.particles(pos, radius=0.01, per_vertex_color=per_color)
    canvas.scene(scene)
    window.show()
