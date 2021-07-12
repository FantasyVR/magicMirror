% 一个ROD系统的仿真 with XPBD
clear;
clc;
MaxSteps = 10000;
MaxIte = 10;
N = 3;
NC = N - 1;
x = zeros(N,2);
for i = 1:N
%    x(i,:) = [0.0,- 0.1 * (i - 1)]; 
     x(i,:) = [0.1 * (i-1), 0.0];
end
v = zeros(N,2);
oldx = x;
mass = zeros(N,1);
for i = 1:N
    mass(i) = 1.0;
end
mass(1) = 0.0;
gravity = [0.0,-9.8];
h = 0.01;
rc = zeros(NC); % reset length

% XPBD variables
compliance = 1.0e-6;
alpha = compliance/h/h;
lambda = zeros(NC,1);

for i = 1:(NC)
    rc(i) = norm(x(i,:) - x(i+1,:));
end

for step = 1:MaxSteps
    %semi-Euler integration
    oldx = x;
    for i = 2:N
        v(i,:) = v(i,:) +  h * gravity;
        if(mass(i) ~= 0.0)
            x(i,:) = x(i,:) + v(i,:) * h;
        end
    end
    %XPBD
    lambda = zeros(NC,1);
    for ite = 1:MaxIte
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
        % 组装系统矩阵
        A = zeros(2*N+NC,2*N+NC);
        for i = 1:N
            A(2*i-1:2*i, 2*i-1:2*i) = mass(i) * eye(2);
        end
        A(2*N+1:end, 1:2*N) = J;
        A(1:2*N, 2*N+1:end) = -J';
        for i = 1:NC
            A(2*N+i,2*N+i) = alpha;
        end
        
        % RHS：b  
        b = zeros(2*N+NC,1);
        b(2*N+1:end) = -c;
       
        
        % Real A & Real b
        A = A(3:end,3:end);
        b = b(3:end);
        
        % 求解系统矩阵
        dx = A \ b;
        % 更新位置
        for i = 1:(N-1)
            x(i+1,:) = x(i+1,:) + dx(2*i-1:2*i)';
        end
        % 更新lambda
        lambda = lambda + dx(2*N-1:end);
    end
    % 更新速度
    for i = 2:N
       v(i,:) = (x(i,:)-oldx(i,:))/h; 
    end
    
end 
