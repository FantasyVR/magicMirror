function [x_pred,x, x_old] = semi_euler(x, mass, v,h,gravity)
%semi_euler
%   motion without constraints
% x_pred : 预测位置
% x：预测位置
% x_old: 上一步的位置
% v: 外力作用下的速度
    N = size(x,1);
    x_old = x;
    for i = 1:N
        v(i,:) = v(i,:) +  h * gravity;
        if(mass(i) ~= 0.0)
            x(i,:) = x(i,:) + v(i,:) * h;
        end
    end
    x_pred = x;
end

