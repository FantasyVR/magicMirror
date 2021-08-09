import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

v1_t0 = np.array([0.5, 0.5])
v1_t1 = np.array([0.6, 0.6])
v2_t0 = np.array([0.2, 0.5])


def vertex_vertexCCD(x_i0, x_i1, x_j0, x_j1, D2):  # From Huamin Wang Paper
    x_ij0 = x_j0 - x_i0
    x_ij1 = x_j1 - x_i1
    v_ji = x_ij1 - x_ij0
    T = np.clip(-np.dot(x_ij0, v_ji) / (np.dot(v_ji, v_ji)), 0, 1)
    cp = x_ij0 + T * v_ji
    if np.dot(cp, cp) <= D2 + 1.0e-6:  # Proximity test
        return True
    else:
        return False


def vertex_vertexCCDA(xi_0, xi_1, xj_0, xj_1,
                      D2):  # From Huamin Wang Source code
    xji = xj_0 - xi_0
    xji_1 = xj_1 - xi_1
    vji = xji_1 - xji
    a = -np.dot(xji, vji)
    b = np.dot(vji, vji)
    if a > 0 and a < b:
        t = a / b
        xji += vji * t
        if np.dot(xji, xji) < D2:
            return True, t
    t = 1
    if np.dot(xji_1, xji_1) < D2:
        return True, t
    return False, t


if __name__ == "__main__":
    gui = ti.GUI("V-V CCD")
    D2 = 0.001
    while gui.running:
        for e in gui.get_events():
            if e.key == gui.ESCAPE:
                gui.running = False
            if e.key == 'a':
                D2 += 0.001
                print(f"D2: {D2}")
            if e.key == 'd':
                D2 -= 0.001
                print(f"D2: {D2}")

        curse_pos = gui.get_cursor_pos()

        gui.line(v1_t0, v1_t1, radius=5, color=0xFFFFFF)

        isInter, t = vertex_vertexCCDA(v1_t0, v1_t1, v2_t0, curse_pos, D2)
        if isInter:
            gui.line(v2_t0, curse_pos, radius=5, color=0xFF0000)
            cp = v2_t0 + t * (curse_pos - v2_t0)
            gui.circle(cp, radius=6, color = 0x0000FF)
        else:
            gui.line(v2_t0, curse_pos, radius=5, color=0x00FF00)
        gui.show()
