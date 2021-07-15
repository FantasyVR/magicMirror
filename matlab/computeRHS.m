function [b] = computeRHS(x, prdx, mass, c, J, lambda)
%UNTITLED11 Summary of this function goes here
%   Detailed explanation goes here
        N = size(mass,1);
        NC = size(lambda,1);
        b = zeros(2*N+NC,1);
        b(2*N+1:end) = -c;
        % Geometric Stiffness
        for i = 1:N
           b(2*i-1:2*i) = -mass(i) * (x(i,:) - prdx(i,:))';
        end
        Gl = J' * lambda;
        b(1:2*N) = b(1:2*N) + Gl; 
end

