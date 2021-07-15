function [v] = updateVel(x,oldx,h)
%UNTITLED13 Summary of this function goes here
%   Detailed explanation goes here
    N = size(x,1);
    for i = 2:N
       v(i,:) = (x(i,:)-oldx(i,:))/h; 
    end
end

