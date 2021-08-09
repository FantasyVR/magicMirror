import taichi as ti
import numpy as np
from vertex_vertex_ccd import vertex_vertexCCD

ti.init(arch=ti.cpu)

v1_t0 = np.array([0.1, 0.2])
v1_t1 = np.array([0.1, 0.6])
v2_t0 = np.array([0.5, 0.3])
v2_t1 = np.array([0.5, 0.4])
h_t0 = np.array([0.2, 0.1])


def vertex_edgeCCD(p0, p1, q0, q1, h0, h1):
    x1, x2 = h0 - p0
    x3, x4 = h0 - q0
    a, b = h1 - h0 - p1 + p0
    c, d = h1 - h0 - q1 + q0
    A = a * d - b * c
    B = d * x1 + a * x4 - c * x2 - b * x3
    C = x1 * x4 - x2 * x3
    print(f"A:{A}, B:{B}, C:{C}")
    if A - 0.0 < 1.0e-10:
        t = -C/B
        p = p0 + t * (p1 - p0)
        q = q0 + t * (q1 - q0)
        h = h0 + t * (h1 - h0)
        if 0 < t < 1 and np.dot(h-p,h-p) <  np.dot(q-p,q-p) and  np.dot(h-q,h-q) <  np.dot(q-p,q-p) :
            print(f"||h-p||: {np.dot(h-p,h-p)}")
            print(f"||q-p||: {np.dot(q-p,q-p)}")
            return True, -C / B
        else:
            return False, -1000.0

    Delta = B * B - 4 * A * C
    print(f"Delta: {Delta}")
    if Delta < 0:
        return False, -1
    t1 = (-B + np.sqrt(Delta)) / (2 * A)
    t2 = (-B - np.sqrt(Delta)) / (2 * A)
    print(f"t1: {t1}, t2: {t2}")
    if 0 < t1 < 1:
        t = -C/B
        p = p0 + t * (p1 - p0)
        q = q0 + t * (q1 - q0)
        h = h0 + t * (h1 - h0)
        if np.dot(h-p,h-p) <  np.dot(q-p,q-p) and  np.dot(h-q,h-q) <  np.dot(q-p,q-p):
            return True, t1
    elif 0 < t2 < 1:
        t = -C/B
        p = p0 + t * (p1 - p0)
        q = q0 + t * (q1 - q0)
        h = h0 + t * (h1 - h0)
        if np.dot(h-p,h-p) <  np.dot(q-p,q-p) and  np.dot(h-q,h-q) <  np.dot(q-p,q-p):
            return True, t2
    return False, -1000.0



if __name__ == "__main__":
    gui = ti.GUI("V-E CCD")
    while gui.running:
        D2 = 0.008
        for e in gui.get_events():
            if e.key == gui.ESCAPE:
                gui.running = False
            if e.key == 'a':
                D2 += 0.001
                print(f"D2: {D2}")

        curse_pos = gui.get_cursor_pos()

        gui.line(v1_t0, v2_t0, radius=5, color=0xFFFFFF)
        gui.line(v1_t1, v2_t1, radius=5, color=0xFFFF00)

        print(f">>> curse_pos: {curse_pos}")
        isInter, t = vertex_edgeCCD(v1_t0, v1_t1, v2_t0, v2_t1, h_t0,
                                    curse_pos)

        if isInter:
            gui.line(h_t0, curse_pos, radius=3, color=0xFF0000)
        else:
            gui.line(h_t0, curse_pos, radius=3, color=0x00FF00)
        cp = h_t0 + t * (curse_pos - h_t0)
        gui.circle(cp, radius=6, color=0x0000FF)

        gui.show()
