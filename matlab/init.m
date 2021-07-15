function [x,oldx,prdx,mass,vel,rc] = init(N, NC)
%UNTITLED14 Summary of this function goes here
%   Detailed explanation goes here
x = zeros(N,2);
oldx = zeros(N,2);
prdx = zeros(N,2);
vel = zeros(N,2);
for i = 1:N
%    x(i,:) = [0.0,- 0.1 * (i - 1)]; 
%      x(i,:) = [0.5 + 0.1 * (i-1), 0.0];
     x(i,:) = [0.1 * (i-1), 0.0];
end

mass = zeros(N,1);
for i = 1:N
    mass(i) = 1.0;
end
mass(1) = 0.0;

rc = zeros(NC,1); % reset length
for i = 1:(NC)
    rc(i) = norm(x(i,:) - x(i+1,:));
end

end

