function [dx, c] = iter(NC,x, prdx,mass,  rc, alpha, lambda)
        % 计算约束
        c = computeConstraint(NC,x,rc,alpha, lambda);
        % 计算Jacobi Matrix
        J = computeJacobianMatrix(NC,x);
        % 计算 Geometric Stiffness Matrix
        K = computeGeometricStiffnessMatrix(x,lambda);
        
        eig_A = eig(K);
        flag = 0;
        for i = 1:rank(K)
            if eig_A(i) <= 0 
                flag = 1;
            end
        end
        
        disp(["K is PSD? : ", flag]);
%         mustBePositive(K);
        % 组装系统矩阵: A
        A = assamble(mass,K,J,alpha);
        % RHS: b  
        b = computeRHS(x,prdx, mass, c, J, lambda);
        
        % Real A & Real b
        A = A(3:end,3:end);
        b = b(3:end);
        disp("norm(b): ");
        disp(norm(b));

        % 求解系统矩阵
        dx = A \ b;
end

