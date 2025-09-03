% 1. Simple initialization with default parameters
addpath("SDK\")
% slm = SLMController();
slm = SLMController('lut_path','C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\LUT Files\slm5691_at635.LUT');
%%
focal_len = 353;
x_mrad = 5;
scale = 0.75;
set_fps = 50;
tic
for i=1:20
    java.lang.Thread.sleep(1/set_fps*1000);
    x_mrad = -x_mrad;
    phase = generateFresnelLensPhase(slm.width,slm.height,focal_len,x_mrad,0,'wavelength', 550e-9);
    % figure;imagesc(phase);colorbar
    slm.uploadImage(phase*scale);
end

disp('Actual fps:')
disp(1/(toc/20))
%% 5. Clean up (happens automatically when slm goes out of scope)
delete(slm); % or just clear slm