from cloth_jacobiAcc import Cloth
import taichi as ti
import numpy as np
import argparse
import os
if __name__ == "__main__":
    """
    With chebyshev solver:     python main_cloth_jacobi_acc.py -o data/no_chebeyshev.txt -u
    Without chebyshev solver:  python main_cloth_jacobi_acc.py -o data/no_chebeyshev.txt
    """
    parser = argparse.ArgumentParser(description='Acc stable constrainted dyanmics')
    parser.add_argument('--output', '-o',nargs=1, default="data.txt", type=str, required=False)
    parser.add_argument('--use-chebyshev', '-u', action='store_true', help='Using checbyshev solver')
    args = parser.parse_args()

    ti.init(arch=ti.cpu)

    output_file = args.output if type(args.output) is str else args.output[0]
    print(args.output)
    if os.path.exists(output_file):
        os.remove(output_file)
    log_file = open(output_file, "w")

    use_chebyshev = args.use_chebyshev
    print(use_chebyshev)

    pause = False
    NumSteps, MaxIte, BreakSteps, dragStep = 1, 20, 500, 600
    gui = ti.GUI('XPBD-FEM')
    cloth = Cloth(10, 0.01)
    steps = 0
    rho	= 0.9992
    omega = 0
    while gui.running:
        for e in gui.get_events():
            if e.key == gui.ESCAPE:
                gui.running = False
            elif e.key == gui.SPACE:
                pause = not pause

        mouse_pos = np.asarray(gui.get_cursor_pos(), dtype=np.float32)
        strength = gui.is_pressed(gui.LMB) - gui.is_pressed(gui.RMB)
        gui.circle(mouse_pos, radius=10, color=0x336699)
        
        if not pause:
            for i in range(NumSteps):
                steps += 1
                log_file.write(f"==============Timestep: {steps}=======================\n")
                print(f"==============Timestep: {steps}=======================", end="\r")
                if steps == dragStep:
                    mouse_pos = np.array([0.5, 0.1], dtype=np.float32)
                    strength = 1
                cloth.semiEuler(mouse_pos, strength)
                cloth.resetLagrangian()
                for ite in range(MaxIte):
                    dual_residual = cloth.computeGradientVector()
                    if ite == MaxIte - 1:
                        log_file.write(f"{np.sqrt(dual_residual)} \n")
                        # print(f"Iteration:{ite+1} \n >>>> Dual residual: {np.sqrt(dual_residual)}")
                    
                    cloth.updatePos()
                    if ite <= 10:
                        omega = 1
                    elif ite == 11:
                        omega = 2/(2-rho * rho)
                        # omega = 0.5
                    else:
                        omega = 4/(4-rho*rho*omega)
                        # omega = 0.5 * omega
                    if use_chebyshev:
                        # cloth.applyPrimalChebyshev(omega)
                        cloth.applyDualChebyshev(omega)


                cloth.updteVelocity()

        if steps==BreakSteps:
            break

        cloth.display(gui)
        gui.show()
    log_file.close()