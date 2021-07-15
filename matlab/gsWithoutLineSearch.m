% 一个ROD系统的仿真 with XPBD + Geometric Stiffness + line search
clear;
clc;
MaxSteps = 1000;
MaxIte = 3;
N = 10;
NC = N - 1;
[x,oldx,prdx,mass,v, rc] = init(N, NC);
gravity = [0.0,-9.8];
h = 0.01;

% XPBD variables
compliance = 1.0e-6;
alpha = compliance/h/h;
lambda = zeros(NC,1);
for step = 1:MaxSteps
    disp(["===========  Start time step", step,"============================="]);
    %semi-Euler integration
    [prdx, x, oldx] = semi_euler(x,mass,v,h,gravity);
    %XPBD
    lambda = zeros(NC,1);
    for ite = 1:MaxIte
        disp(["*************Start iteration", ite,"****************"]);
        [dx, c] = iter(NC,x, prdx,mass,  rc, alpha, lambda);
        disp("norm(dx): ");
        disp(norm(dx));
        % line search
        lsSize = 1.0; %line search step size
        [xx, ll] = updatePosLambda(x,lambda,dx, lsSize);
        x = xx;
        lambda = ll;
        disp("norm(lambda): ");
        disp(norm(lambda));
    end
    % 更新速度
    v = updateVel(x,oldx,h);
end 
