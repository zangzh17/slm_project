 function [T,x] = SFR1D(t,z,pitch,lam)
% calculate single-FT-based Fresnel Transform (SFR)
%
% % INPUT PARAMETERS
% % t: input field (near square)
% % z: propagation distance
% % pitch: sampling interval
% % lam: wavelength
%% parameters
t = t(:).';
dx = pitch;
N = length(t);
k = 2*pi/lam;
%% critical range
zc = N*dx^2/lam;
if z<zc
    warning(['Range ',num2str(z*1e3,1),' mm is less than Zc = ',num2str(zc*1e3,1),' mm'])
end
%% inner chirp
lx = N*dx;
x0 = linspace(-lx/2,lx/2-lx/N,N);
chirp0 = exp(1i*k/2/z*(x0.^2));
t = t.*chirp0;
%% sampling number of output field
N2 = floor(lam*z/dx^2-N);
N2 = max([N,N2]);
%% FT
pad_x = ceil((N2-N)/2);
t_pro = fftshift(fft(fftshift(padarray(t,[0,pad_x])))) * dx;
N2 = length(t_pro); % update
%% outer chirp
Lx = lam*z/dx;
xn = linspace(-Lx/2,Lx/2-Lx/N2,N2);
Chirpn = exp(1i*k/2/z*(xn.^2) + 1i*k*z);
t_pro = Chirpn.*t_pro / (1i*lam*z);

%% Clip to get central range result
% LLx = lam*z/dx-Nx*dx;
% if LLx<=0 
%     warning('No output due to too small range')
% end
% T = t_pro(abs(xn)<=LLx/2);
% Nx2 = length(T); % update
% x = linspace(-LLx/2,LLx/2-LLx/Nx2,Nx2);

%% Non-clip
pitch2 = lam*z/pitch/N2;
T = t_pro * sqrt(pitch2 * pitch2 * N2);
x = xn;