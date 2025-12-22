%% Use loaded ATF
% Load pupil func etc.
load('../atf.mat');
% load('../atf_fresnel.mat');

% convert
atf = permute(atf,[2,3,1]);

% show
figure;
for i=1:M
    for j=1:M
        index = sub2ind([M,M],i,j);
        imagesc(abs(atf(:,:,index)));colormap("gray");
        title(sprintf('Lens: (%d,%d)',i,j))
        drawnow
        pause(0.05)
    end
end

