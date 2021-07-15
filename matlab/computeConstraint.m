function [c] = computeConstraint(NC, x, rc, alpha, lambda)
% 计算约束violation
        % 计算约束
        c = zeros(NC,1); % constraint violation
        for i = 1 : (NC)
            c(i) = norm(x(i,:) - x(i+1,:)) - rc(i);
        end
        % XPBD
        c = c + alpha * lambda; 
end

