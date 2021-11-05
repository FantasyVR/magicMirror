from cloth import Cloth
import taichi as ti
import numpy as np
if __name__ == "__main__":
    ti.init(arch=ti.cpu)

    pause = False
    NumSteps, MaxIte = 3, 4

    gui = ti.GUI('XPBD-FEM')
    cloth = Cloth(10, 0.01)
    while gui.running:
        for e in gui.get_events():
            if e.key == gui.ESCAPE:
                gui.running = False
            elif e.key == gui.SPACE:
                pause = not pause

        mouse_pos = gui.get_cursor_pos()
        strength = gui.is_pressed(gui.LMB) - gui.is_pressed(gui.RMB)
        gui.circle(mouse_pos, radius=15, color=0x336699)
        
        if not pause:
            for i in range(NumSteps):
                cloth.semiEuler(np.asarray(mouse_pos), strength)
                cloth.resetLagrangian()
                for ite in range(MaxIte):
                    cloth.solveConstraints()
                cloth.updteVelocity()

        cloth.display(gui)
        gui.show()