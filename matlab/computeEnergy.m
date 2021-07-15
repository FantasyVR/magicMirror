function [e] = computeEnergy(x,prdx,mass,rc,NC,lambda,alpha)
% 计算系统能量
    N = size(x,1);
    e = 0.0;
    for i = 1: N
       dx = x(i,:) - prdx(i,:);
       e = e + mass(i) * dot(dx,dx);
    end
    constraint = computeConstraint(NC, x, rc, alpha, lambda);
    for i = 1 : NC
       e = e + alpha * constraint(i)^2 ;
    end
end

