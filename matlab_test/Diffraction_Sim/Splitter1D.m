function [x] = Splitter1D(M,orders,DoeParm)
% DOE splitter params.
theta = 9.3;
runTimes = 500;
N_step = 255;
lambda = 1.55e-3;
% Fourier orders
N = round(M*1.5);
if mod(M,2)~=mod(N,2), N=N+1; end
df = 2*sind(theta)/(M-1)/lambda;
% spatial sampling
pitch = 1/df/N;
% Fourier order num
order_shift = (N-M)/2;
order_ind = orders+order_shift;
% Optimization
phi = Random1D(N,N_step,order_ind,runTimes);
x = phi/2/pi;
x = x(:)';
% convolution with pixel unit
pitchNum = (pitch*1e3/DoeParm.pixelSize);
disp(['Pitch = ' num2str(pitch*1e3) ' um'])
disp(['      = ' num2str(pitchNum) ' Pixels'])
x = kron(x,ones(1,round(pitchNum)));
end