function [target,z_target] = target_edof(N, pitch, lam, z, F_obj, dof_factor, N_depth)
% output: [num_depth,N]
% 2 focal points
focal_pos = [ceil((1+N/2)/2),ceil((1+3*N/2)/2)];

% diffraction limit
M = N / 2; % half eff. aperture size
sigma = 0.84*(z * lam) / (M * pitch)/pitch;
fprintf('Img. Spot FWHM: %.1f um\n', sigma*pitch/1.699*1e6);
% diffraction limited depth of field
na = (M*pitch/2) / F_obj; % sub-aperture NA
dof_obj = lam/(na^2) * dof_factor; % objective space DOF
dof = dof_obj * (z/F_obj)^2; % image space DOF
fprintf('sub-aperture NA: %.3f\n', na);
fprintf('Rayleigh Resol.: %.1f um\n', 0.61*lam/na * 1e6);
fprintf('Original Obj. DOF: %.1f um\n', dof_obj/dof_factor*1e6);
fprintf('Enhanced Obj. DOF: %.1f um\n', dof_obj*1e6);
fprintf('EDOF Factor: %.1f \n', dof_factor);
fprintf('Original Img. DOF: %.1f um\n', dof/dof_factor*1e6);
fprintf('Enhanced Img. DOF: %.1f um\n', dof*1e6);
% init
target = zeros(N_depth,N);
% target intensities
for k = 1:size(target,1)
    for i = 1:length(focal_pos)
        x = 1:N;
        gauss = exp(-2*((x - focal_pos(i)).^2) / (sigma^2));
        target(k,:) = target(1,:) + gauss;
    end
    target(k,:) = target(k,:)/sum(target(k,:))*N;
end
if N_depth>1
    z_target = z + linspace(-dof/2, dof/2, N_depth);
else
    z_target = z;
end
end