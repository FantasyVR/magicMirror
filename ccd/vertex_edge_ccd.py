import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

v1_t0 = np.array([0.2, 0.2])
v1_t1 = np.array([0.2, 0.6])
v2_t0 = np.array([0.6, 0.3])
v2_t1 = np.array([0.6, 0.4])
h_t0 = np.array([0.3, 0.3])


def isOnSegment(p0, p1, q0, q1, h0, h1, t):
    p = p0 + t * (p1 - p0)
    q = q0 + t * (q1 - q0)
    h = h0 + t * (h1 - h0)
    hp = h - p
    hq = h - q
    pq = p - q
    isInter = True
    if t < 0 or t > 1 or np.dot(hp, hp) > np.dot(pq, pq) or np.dot(
            hq, hq) > np.dot(pq, pq):
        isInter = False
    return isInter


def Quadratic_Solver(a,b,c):
    root = []
    if a < 0:
        a, b, c = -a, -b, -c
    delta = b*b - 4*a*c
    if delta <= 0:
        if -b > 0 and -b<2*a:
            root.append(-b/2*a)
            return root 
    
    if b <= 0:
        tmp = -b + np.sqrt(delta)
        twice_c = 2 * c
        twice_a = 2 * a
        if twice_c>0 and twice_c < tmp:
            root.append(twice_c/tmp)
        if tmp < twice_a:
            root.append(tmp/twice_a)
    else:
        tmp = -b-np.sqrt(delta)
        twice_c = 2*c
        twice_a = 2*a
        if twice_a < tmp:
            root.append(tmp/twice_a)
        if twice_c < 0 and twice_c > tmp:
            root.append(twice_c/tmp)
    return root


def vertex_edgeCCD(p0, p1, q0, q1, h0, h1):
    x1, x2 = h0 - p0
    x3, x4 = h0 - q0
    a, b = h1 - h0 - p1 + p0
    c, d = h1 - h0 - q1 + q0
    A = a * d - b * c
    B = d * x1 + a * x4 - c * x2 - b * x3
    C = x1 * x4 - x2 * x3
    print(f"A:{A}, B:{B}, C:{C}")
    # roots = Quadratic_Solver(A,B,C)
    # print(f"len(roots): {len(roots)}, roots: {roots}")
    # return len(roots)>0, roots[0]

    t = -1.0
    isInter = False
    if A - 0.0 < 1.0e-10:
        t = -C / B
        isInter = isOnSegment(p0, p1, q0, q1, h0, h1, t)
    Delta = B * B - 4 * A * C
    print(f"Delta: {Delta}")
    if not isInter and Delta >= 0:
        t1 = (-B + np.sqrt(Delta)) / (2 * A)
        t2 = (-B - np.sqrt(Delta)) / (2 * A)
        print(f"t1: {t1}, t2: {t2}")
        if 0.0 <= t1 <= 1.0 and 0.0 <= t2 <= 1.0:
            t = min(t1, t2)
            print(t)
            isInter = isOnSegment(p0, p1, q0, q1, h0, h1, t)
        elif 0.0 <= t1 <= 1.0:
            isInter = isOnSegment(p0, p1, q0, q1, h0, h1, t1)
            t = t1
        elif 0 <= t2 <= 1.0:
            isInter = isOnSegment(p0, p1, q0, q1, h0, h1, t2)
            t = t2
    else:
        t = -B/(2*A)
        isInter = isOnSegment(p0, p1, q0, q1, h0, h1, t)
    return isInter, t


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
            cp = h_t0 + t * (curse_pos - h_t0)
            gui.circle(cp, radius=6, color=0x0000FF)
        else:
            gui.line(h_t0, curse_pos, radius=3, color=0x00FF00)
        gui.show()
