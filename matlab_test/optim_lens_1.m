N = 1000;
pitch = 10e-6;
D = N*pitch;
lam = 0.632e-6;
z = 50e-3;
n_iter = 200;
addpath('Diffraction_Sim\')

% ASM kernel
[H,N_pad] = ASM_Kernel(z,pitch,lam,N);

% defousing initial wavefront
F_obj = 10e-3; % focal length of obj lens
na_obj = 0.05; % NA of obj lens (divided)
dz = 1.0 * lam/(2*na_obj^2); % target defocus (relative to DOF)
F = 1/(1/F_obj - 1/(dz+F_obj)) % effective image axial shift

% effective disparity due to defocus and aperture distance
x_dp = D/2*z /F / pitch
y_i = spherical_wave(F, N_pad, pitch, lam); % define wavefront


% target 1
sigma = 0.84*(z * lam) / (N * pitch)/pitch; % diffraction limit
positions = [round(N/4), round(3*N/4)];
% positions = [1, N];
target = zeros(2,N);
for i = 1:length(positions)
    pos = positions(i);
    x = 1:N;
    gauss = exp(-2*((x - pos).^2) / (sigma^2));
    target(1,:) = target(1,:) + gauss;
end
target(1,:) = target(1,:)/sum(target(1,:))*N;
% target 2
for i = 1:length(positions)
    pos = positions(i);
    pos = pos + x_dp * sign(pos-N/2); 
    x = 1:N;
    gauss = exp(-2*((x - pos).^2) / (sigma^2));
    target(2,:) = target(2,:) + gauss;
end
target(2,:) = target(2,:)/sum(target(2,:))*N;

% initial value (normalized)
phi = 2*pi*rand(1,N);
N_step = 255;
phi_check = phi;
dphi = 2*pi/N_step;

% use GPU
H = gpuArray(single(H));
phi = gpuArray(single(phi));
y_i = gpuArray(single(y_i));
% define loss function
loss_func = @(x) loss(x,H,y_i,target);

figure
tic
for ii=1:n_iter
    ind = randperm(N);
    for jj=1:N
        f0 = loss_func(phi);
        phi(ind(jj)) = mod(phi(ind(jj))+dphi, 2*pi);
        f = loss_func(phi);
        if f>f0
            % worse, go backwards
            phi(ind(jj)) = mod(phi(ind(jj))-2*dphi, 2*pi);
            f = loss_func(phi);
            if f>f0
                % worse, go backwards
                phi(ind(jj)) = mod(phi(ind(jj))+dphi, 2*pi);
            end
        end
    end
    % show progress
    subplot(211)
    I = forward(phi,H,y_i);
    plot(I(1,:),'Color',[0.8500,0.35,0.15]); hold on
    plot(target(1,:),'--r');
    plot(I(2,:),'Color',[0,0.52,0.7410]);
    plot(target(2,:),'--b');
    hold off
    subplot(212)
    plot(phi)
    title(sprintf( 'Loop: %d  Loss:%.2e  Avg time:%.1f sec', ii, f0, toc/ii ))
    drawnow
    % check if converge
    if sum(abs(phi_check-phi))<dphi/2
        break
    else
        phi_check = phi;
    end
end

function output = loss(x,H,y_i,target)
I = forward(x,H,y_i);
output = sum((I-target).^2,"all");
end

function output = forward(x,H,y_i)
y1 = prop(exp(1j*x),H,'circular');
y2 = prop(exp(1j*x),H,'circular',y_i);
output(1,:) = abs(y1).^2;
output(2,:) = abs(y2).^2;
end

z_list = linspace(0,z*1.2,1000);
z_idx = find(z>=z_list,1,"last");
I_stack = zeros(length(z_list),N);
for ii=1:length(z_list)
    y = ASM1D(exp(1j*phi),z_list(ii),pitch,lam);
    % disp(sum(abs(y).^2))
    I = abs(y).^2;
    I_stack(ii,:) = I;
end
% figure
% imagesc(I_stack)

I_stack2 = zeros(length(z_list),N);
F = 2;
y_i = spherical_wave(F, N, pitch, lam);
for ii=1:length(z_list)
    y = ASM1D(y_i.*exp(1j*phi),z_list(ii),pitch,lam);
    I = abs(y).^2;
    I_stack2(ii,:) = I;
end
figure('Position',[100 100 400 200])
subplot(121)
imagesc(I_stack)
subplot(122)
imagesc(I_stack2)

figure('Position',[100 100 500 200])
I = forward(phi,z,pitch,lam,F);
plot(I(1,:),'linewidth',1,'Color',[0.8500,0.35,0.15]); hold on
plot(target(1,:),'--r');
plot(I(2,:),'linewidth',1,'Color',[0,0.52,0.7410]);
plot(target(2,:),'--b');


