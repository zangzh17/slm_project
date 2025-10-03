function [dx,df] = SamplingParm(theta_m,lam,Nx,Nfx)
df = 2*sind(theta_m)/(Nfx-1)/lam;
dx = 1/df/Nx; 
disp(['Pitch (um): ' num2str(dx*1e3)])
end