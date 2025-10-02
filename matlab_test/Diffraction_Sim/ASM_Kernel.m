function [H,Nx_pad] = ASM_Kernel(z_list,pitch,lam,Nx)
k = 2*pi/lam;
zc = critical_range(pitch,lam,Nx);
% padding
pad_x = ceil(Nx/2);
Nx_pad = Nx + pad_x*2;
% create transfer function
H = zeros(length(z_list), Nx_pad);
for i=1:length(z_list)
    z = z_list(i);
    if abs(z)>zc
        % option 2 RSC
        lbx = Nx_pad*pitch;
        xb = linspace(-lbx/2,lbx/2-pitch,Nx_pad);
        R = sqrt(xb.^2+z^2);
        h = 1/2/pi * abs(z)./R .* (1./R-sign(z)*1i.*k) .* exp(sign(z)*1i*k*R)./R;
        H(i,:) = fft(fftshift(h));
        H(i,:) = fftshift(H(i,:)/abs(H(i,1)));
    else
        % ASM
        flbx = 1/pitch;
        dfx = flbx/Nx_pad;
        fbx = linspace(-flbx/2,flbx/2-dfx,Nx_pad);
        H(i,:) = exp(1i*k*z*sqrt(1-(lam*fbx).^2));
    end
end