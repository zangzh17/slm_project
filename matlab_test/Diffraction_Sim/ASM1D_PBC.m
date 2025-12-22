function [T,x] = ASM1D_PBC(t,z,pitch,lam,ind_focal_length)
% calculate Angular Spectrum Method (ASM) or Rayleigh-Sommerfield Convolution (RSC)
% use Periodic boundary conditions (PBC)
% according to propogation distance z
% z>zc for RSC; z<=zc for ASM
%
% % INPUT PARAMETERS
% % t: input field (near square)
% % z: propagation distance
% % pitch: sampling interval
% % lam: wavelength
%% parameters
t = t(:).';
dx = pitch;
Nx = length(t);
k = 2*pi/lam;
x = linspace(-Nx*dx/2,Nx*dx/2-dx,Nx);
%% padding 
pad_x = ceil(Nx/2);
t_pad = padarray(t,[0,pad_x],"circular");
Nx_pad = Nx + pad_x*2;
%% set incident spherical wave & fft
if nargin==5
    yi = spherical_wave(ind_focal_length, Nx_pad, pitch, lam);
    t_pad = t_pad.*yi;
end
t_pad_FT = fftshift(fft(fftshift(t_pad)));
%% choose method
zc = 2*Nx*dx^2/lam*sqrt(1-(lam/(2*dx))^2);
%% calc ASM/RSC
if abs(z)>zc
    % option 2 RSC
    % impulse response
    lbx = Nx_pad*dx;
    xb = linspace(-lbx/2,lbx/2-dx,Nx_pad);
    R = sqrt(xb.^2+z^2);
    h = 1/2/pi * abs(z)./R .* (1./R-sign(z)*1i.*k) .* exp(sign(z)*1i*k*R)./R;
    % convolution
    H = fft(fftshift(h));
    H = fftshift(H/abs(H(1)));
else
    % ASM
    % transfer function
    flbx = 1/pitch;
    dfx = flbx/Nx_pad;
    fbx = linspace(-flbx/2,flbx/2-dfx,Nx_pad);
    H = exp(1i*k*z*sqrt(1-(lam*fbx).^2));
end
t_pro = ifftshift(ifft(ifftshift(t_pad_FT.*H)));
T = t_pro(pad_x+1:pad_x+Nx);
end
