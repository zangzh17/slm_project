function [fx,fy] = FourierOrder(sz,pitch)
% % INPUT PARAMETERS
% % sz: vector in real space
% % pitch: sampling interval/um
if length(sz)==1
    sz = [1,sz];
end
Nx = sz(2); Ny = sz(1);
if nargin<2
    dfx = 1; dfy = 1;
else
    dx = pitch; dy = pitch;
    flbx = 1/dx; flby = 1/dy;
    dfx = flbx/Nx; dfy = flby/Ny;
end
fx = (-floor(Nx/2):1:ceil(Nx/2)-1)*dfx;
fy = (-floor(Ny/2):1:ceil(Ny/2)-1)*dfy;
end