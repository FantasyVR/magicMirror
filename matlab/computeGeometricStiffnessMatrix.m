function [K] = computeGeometricStiffnessMatrix(x, lambda)
% 计算Geometric stiffness matrix
    N = size(x,1);
    NC = size(lambda,1);
    K = zeros(2*N,2*N);
        for i = 1 : NC
            e = x(i,:)-x(i+1,:);
            n = e/norm(e);
            k = lambda(i)/norm(e) * (eye(2)- n' * n);
            K(2*i-1:2*i,2*i-1:2*i)      = K(2*i-1:2*i,2*i-1:2*i)      + k;
            K(2*i-1:2*i,2*i+1:2*i+2)    = K(2*i-1:2*i,2*i+1:2*i+2)    - k;
            K(2*i+1:2*i+2, 2*i+1:2*i+2) = K(2*i+1:2*i+2, 2*i+1:2*i+2) + k;
            K(2*i+1:2*i+2,2*i-1:2*i)    = K(2*i+1:2*i+2,2*i-1:2*i)    - k;
%             disp(["k", i]);
%             disp(k);
        end
end

