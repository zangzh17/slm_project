N = 200;
f = @() generateFresnelLensPhase_mex(1920,1152,353,5,0,1,550e-9,9.2e-6);
g = @() generateFresnelLensPhase(1920,1152,353,5,0,1,550e-9,9.2e-6);

time_mex = timeit(f)*1000;
time_m   = timeit(g)*1000;


disp([time_mex, time_m])  % [ms per call]


phase_mex = generateFresnelLensPhase_mex(1920,1152,353,5,0,1,550e-9,9.2e-6);
phase_matlab = generateFresnelLensPhase(1920,1152,353,5,0,1,550e-9,9.2e-6);

figure
nexttile
imagesc(phase_matlab)
colorbar
nexttile
imagesc(phase_mex)
colorbar