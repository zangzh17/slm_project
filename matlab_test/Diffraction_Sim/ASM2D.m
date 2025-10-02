function [T,x,y] = ASM2D(t,z,pitch,lam)
% calculate Angular Spectrum Method (ASM) or Rayleigh-Sommerfield Convolution (RSC)
% according to propogation distance z
% z>zc for RSC; z<=zc for ASM
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
x = linspace(-Nx*dx/2,Nx*dx/2-dx,Nx);
y = linspace(-Ny*dy/2,Ny*dy/2-dy,Ny);
%% choose method
zcx = 2*Nx*dx^2/lam*sqrt(1-(lam/(2*dx))^2);
zcy = 2*Ny*dy^2/lam*sqrt(1-(lam/(2*dy))^2);
zc = (zcx+zcy)/2;
%% calc ASM/RSC
if abs(z)>zc
    % RSC
    % n0-point zero-padded input field
    pad_x = ceil(Nx/2);
    pad_y = ceil(Ny/2);
    t_pad = padarray(t,[pad_y,pad_x]);
    t_pad_FT = fftshift(fft2(fftshift(t_pad)));
    Nx_pad = size(t_pad,2);
    Ny_pad = size(t_pad,1);
    lbx = Nx_pad*dx;
    lby = Ny_pad*dy;
    xb = linspace(-lbx/2,lbx/2-dx,Nx_pad)';
    yb = linspace(-lby/2,lby/2-dy,Ny_pad)';
    [X,Y] = meshgrid(xb,yb);
    % option 2 RSC
    R = sqrt(X.^2+Y.^2+z^2);
    h1 = 1/2/pi*z./R.*(1./R-1i*k).*exp(1i*k*R)./R;
    H1 = fftshift(fft2(fftshift(h1)));
    t_pro = ifftshift(ifft2(ifftshift(t_pad_FT.*H1)));
    T = t_pro(pad_y+1:pad_y+Ny, pad_x+1:pad_x+Nx);
else
    % ASM
    % zero padding
    pad_x = round(lam*abs(z)/2/dx^2/sqrt(1-(lam/2/dx)^2));
    pad_y = round(lam*abs(z)/2/dy^2/sqrt(1-(lam/2/dy)^2));
    pad_x = ceil(pad_x/2);
    pad_y = ceil(pad_y/2);
    t_pad = padarray(t,[pad_y,pad_x]);
    Nx_pad = size(t_pad,2);
    Ny_pad = size(t_pad,1);
    % transfer function
    flbx = 1/pitch;
    flby = 1/pitch;
    dfx = flbx/Nx_pad;
    dfy = flby/Ny_pad;
    fbx = linspace(-flbx/2,flbx/2-dfx,Nx_pad)';
    fby = linspace(-flby/2,flby/2-dfy,Ny_pad)';
    [fx,fy] = meshgrid(fbx,fby);
    H = exp(1i*k*z*sqrt(1-(lam*fx).^2-(lam*fy).^2));
    t_pad_FT = fftshift(fft2(fftshift(t_pad)));
    t_pro = ifftshift(ifft2(ifftshift(t_pad_FT.*H)));
    T = t_pro(pad_y+1:pad_y+Ny, pad_x+1:pad_x+Nx);
end
end
