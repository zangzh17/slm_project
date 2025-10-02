 function [T,x,y] = SFR2D(t,z,pitch,lam)
% calculate single-FT-based Fresnel Transform (SFR)
%
% % INPUT PARAMETERS
% % t: input field (near square)
% % z: propagation distance/mm
% % pitch: sampling interval/mm
% % lam: wavelength/mm
%% parameters
dx = pitch; dy = dx;
Nx = size(t,2); Ny = size(t,1);
k = 2*pi/lam;
%% critical range
zcx = Nx*dx^2/lam;
zcy = Ny*dy^2/lam;
zc = max([zcx,zcy]);
if z<zc
    warning(['Range ',num2str(z,3),' mm is less than Zc = ',num2str(zc,3),' mm'])
end
%% inner chirp
lx = Nx*dx;
ly = Ny*dy;
x0 = linspace(-lx/2,lx/2-lx/Nx,Nx)';
y0 = linspace(-ly/2,ly/2-ly/Ny,Ny)';
[X0,Y0] = meshgrid(x0,y0);
chirp0 = exp(1i*k/2/z*(X0.^2+Y0.^2));
t_c0 = t.*chirp0;
%% sampling number of output field
Nx2 = floor(lam*z/dx^2-Nx);
Ny2 = floor(lam*z/dy^2-Ny);
Nx2 = max([Nx,Nx2]);
Ny2 = max([Ny,Ny2]);
%% FT
pad_x = ceil((Nx2-Nx)/2);
pad_y = ceil((Ny2-Ny)/2);
t_pro = fftshift(fft2(fftshift(padarray(t,[pad_y,pad_x]))));
Nx2 = size(t_pro,2); % update
Ny2 = size(t_pro,1);
%% outer chirp
Lx = lam*z/dx;
Ly = lam*z/dy;
xn = linspace(-Lx/2,Lx/2-Lx/Nx2,Nx2)';
yn = linspace(-Ly/2,Ly/2-Ly/Ny2,Ny2)';
[Xn,Yn] = meshgrid(xn,yn);
Chirpn = exp(1i*k/2/z*(Xn.^2+Yn.^2));
t_pro = Chirpn.*t_pro;

%% Clip to get central range result
LLx = lam*z/dx-Nx*dx;
LLy = lam*z/dy-Ny*dy;
if LLx<=0 || LLy<=0
    warning('No output due to too small range')
end
T = t_pro(abs(yn)<=LLy/2, abs(xn)<=LLx/2);
Nx2 = size(T,2); % update
Ny2 = size(T,1);
x = linspace(-LLx/2,LLx/2-LLx/Nx2,Nx2);
y = linspace(-LLy/2,LLy/2-LLy/Ny2,Ny2);