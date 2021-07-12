% 一个ROD系统的仿真 with XPBD + Geometric Stiffness
clear;
clc;
MaxSteps = 21;
MaxIte = 11;
N = 3;
NC = N - 1;
x = zeros(N,2);
for i = 1:N
%    x(i,:) = [0.0,- 0.1 * (i - 1)]; 
%      x(i,:) = [0.5 + 0.1 * (i-1), 0.0];
     x(i,:) = [0.1 * (i-1), 0.0];
end
v = zeros(N,2);
oldx = x;
prdx = x;
mass = zeros(N,1);
for i = 1:N
    mass(i) = 1.0;
end
mass(1) = 0.0;
gravity = [0.0,-9.8];
h = 0.01;
rc = zeros(NC,1); % reset length

% XPBD variables
compliance = 1.0e-6;
alpha = compliance/h/h;
lambda = zeros(NC,1);

for i = 1:(NC)
    rc(i) = norm(x(i,:) - x(i+1,:));
end

for step = 1:MaxSteps
    disp(["===========  Start time step", step,"============================="]);
    %semi-Euler integration
    oldx = x;
    for i = 2:N
        v(i,:) = v(i,:) +  h * gravity;
        if(mass(i) ~= 0.0)
            x(i,:) = x(i,:) + v(i,:) * h;
        end
    end
    prdx = x;
    
    %XPBD
    lambda = zeros(NC,1);
    for ite = 1:MaxIte
        disp(["*************Start iteration", ite,"****************"]);
        disp("position: ");
        disp(reshape(x',6,1)');
        % 计算约束
        c = zeros(NC,1); % constraint violation
        for i = 1 : (NC)
            c(i) = norm(x(i,:) - x(i+1,:)) - rc(i);
        end
        % XPBD
        c = c + alpha * lambda; 
        % 计算Jacobi Matrix
        J = zeros(NC, 2 * N);
        for i = 1 : NC
            g1 = (x(i,:) - x(i+1,:))/ norm(x(i,:) - x(i+1,:));
            J(i,2*i-1:2*i)   =  g1;
            J(i,2*i+1:2*i+2) = -g1;
        end
        % 计算 Geometric Stiffness Matrix
        K = zeros(2*N,2*N);
        for i = 1 : NC
            e = x(i,:)-x(i+1,:);
            squareLen = dot(e,e);
            k = - lambda(i)/sqrt(squareLen) * (eye(2)- e' * e/squareLen);
            K(2*i-1:2*i,2*i-1:2*i)      = K(2*i-1:2*i,2*i-1:2*i)      - k;
            K(2*i-1:2*i,2*i+1:2*i+2)    = K(2*i-1:2*i,2*i+1:2*i+2)    + k;
            K(2*i+1:2*i+2, 2*i+1:2*i+2) = K(2*i+1:2*i+2, 2*i+1:2*i+2) -k;
            K(2*i+1:2*i+2,2*i-1:2*i)    = K(2*i+1:2*i+2,2*i-1:2*i)    + k;
        end
        disp("K:");
        disp(K);
        % 组装系统矩阵
        A = zeros(2*N+NC,2*N+NC);
        for i = 1:N
            A(2*i-1:2*i, 2*i-1:2*i) = mass(i) * eye(2);
        end
        % geometric stiffness matrix
        A(1:2*N,1:2*N) = A(1:2*N,1:2*N)+K;
        
        A(2*N+1:end, 1:2*N) = J;
        A(1:2*N, 2*N+1:end) = -J';
        for i = 1:NC
            A(2*N+i,2*N+i) = alpha;
        end
        
        % RHS：b  
        b = zeros(2*N+NC,1);
        b(2*N+1:end) = -c;
        % Geometric Stiffness
        for i = 1:N
           b(2*i-1:2*i) = -mass(i) * (x(i,:) - prdx(i,:))';
        end
        b(1:2*N) = b(1:2*N) + J' * lambda;
        
        disp("Jacobian: ");
        disp(J);
        disp("GLambda:");
        disp((J' *lambda)');
        disp("A: ");
        disp(A);
        disp("b: ");
        disp(b');
        % Real A & Real b
        A = A(3:end,3:end);
        b = b(3:end);
        
        % 求解系统矩阵
        dx = A \ b;
        disp("Solusion: ");
        disp(dx');
        % 更新位置
        for i = 1:(N-1)
            x(i+1,:) = x(i+1,:) + dx(2*i-1:2*i)';
        end
        disp("Corrected position");
        disp(reshape(x',6,1)');
        % 更新lambda
        lambda = lambda + dx(2*N-1:end);
        disp("Corrected lambda: ");
        disp(lambda');
    end
    % 更新速度
    for i = 2:N
       v(i,:) = (x(i,:)-oldx(i,:))/h; 
    end
    disp("Old position");
    disp(reshape(oldx',6,1)');
    disp("velocities: ");
    disp(reshape(v',6,1)');
end 
