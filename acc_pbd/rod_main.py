from rod import Rod
import taichi as ti
import os
import argparse

count = 0
rho	= 0.9992
omega = 0
def step(r, NStep, NMaxIte, log_file, solver_type=0):
    global count, rho, omega
    for step in range(NStep):
        log_file.write(f"######################### Timestep: {count} ##############################\n")
        count += 1
        r.semiEuler()
        for ite in range(NMaxIte):
            log_file.write(f"------------------  iteration: {ite} --------------\n")
            r.resetK()
            r.computeCg()
            r.assemble(r.mass.to_numpy(), r.pos.to_numpy(),
                        r.predictionPos.to_numpy(), r.gradient.to_numpy(),
                        r.K.to_numpy(), r.lagrangian.to_numpy(),
                        r.constraint.to_numpy(), r.disConsIdx.to_numpy())
            dx = r.solve(solver_type)
        
            log_file.write(f">>> Primal Residual: {r.primal_residual} \n")
            log_file.write(f">>> Dual Residual: {r.dual_residual} \n")

            if ite <= 10:
                omega = 1
            elif ite == 11:
                omega = 2/(2-rho * rho)
            else:
                omega = 4/(4-rho*rho*omega)
            r.updatePosLambda(dx)
            # r.applyChebyshev(omega)

        r.updateV()
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Acc stable constrainted dyanmics')
    parser.add_argument('--solver_type','-s',nargs=1, default=0, type=int, choices=[0,1], required=False)
    parser.add_argument('--config_amgx', '-c', nargs=1, default="config/agg_jacobi.json", type=str, required=False)
    parser.add_argument('--output', '-o',nargs=1, default="data/data.txt", type=str, required=False)
    args = parser.parse_args()

    ti.init(arch=ti.cpu)
    output_file = args.output if type(args.output) is str else args.output[0]
    print(args.output)
    if os.path.exists(output_file):
        os.remove(output_file)
    log_file = open(output_file, "w")

    rods = Rod(1000,0.01)
    rods.initRod()
    rods.initConstraint()
    amg_conf_file = args.config_amgx if type(args.config_amgx) is str else args.config_amgx[0]
    rods.init_amgx(amg_conf_file)
    NStep, NMaxIte = 1, 15  # number of steps in each frame, number of iterations in each step

    gui = ti.GUI('AMGX + Stable Constrainted Dynamics v4')
    pause = True
    frame = 0
    solver_type = args.solver_type if type(args.solver_type) is int else args.solver_type[0]
    while gui.running:
        for e in gui.get_events(gui.PRESS):
            if e.key == gui.ESCAPE:
                gui.running = False
            if e.key == gui.SPACE:
                pause = not pause
        if not pause:
            step(rods, NStep, NMaxIte, log_file, solver_type)
        rods.display(gui)    
        if frame == 300:
            break
        frame+=1
        gui.show()
    
    log_file.close()
    rods.detroy_amgx()