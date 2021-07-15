function [J] = computeJacobianMatrix(NC, x)
% 计算Jacobian matix
        N = size(x,1);
        J = zeros(NC, 2 * N);
        for i = 1 : NC
            g1 = (x(i,:) - x(i+1,:))/ norm(x(i,:) - x(i+1,:));
            J(i,2*i-1:2*i)   =  g1;
            J(i,2*i+1:2*i+2) = -g1;
        end
end

