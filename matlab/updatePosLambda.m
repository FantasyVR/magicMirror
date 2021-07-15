function [newX,l] = updatePosLambda(x,lambda,dx, step)
% 根据系统求解，更新位置和拉格朗日橙子
        % 更新位置
        N = size(x,1);
        newX = x;
        for i = 1:(N-1)
            newX(i+1,:) = x(i+1,:) + step * dx(2*i-1:2*i)';
        end
        % 更新lambda
        l = lambda + step * dx(2*N-1:end);
end

