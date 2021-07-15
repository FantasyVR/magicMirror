function [A] = assamble(mass,K,J,alpha)
% 组装系统矩阵
N = size(mass,1);
NC = size(J,1);
A = zeros(2*N+NC,2*N+NC);
        for i = 1:N
            A(2*i-1:2*i, 2*i-1:2*i) = mass(i) * eye(2);
        end
        % geometric stiffness matrix
        A(1:2*N,1:2*N) = A(1:2*N,1:2*N) - K;
        
        A(2*N+1:end, 1:2*N) = J;
        A(1:2*N, 2*N+1:end) = -J';
        for i = 1:NC
            A(2*N+i,2*N+i) = alpha;
        end
end

