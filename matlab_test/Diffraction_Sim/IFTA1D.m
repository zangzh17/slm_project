function phi = IFTA1D(N,orders)
% order bool index
index = false(1,N);
index(orders) = true;
% initial value (normalized)
phi0 = 2*pi*rand(1,N);
t0 = exp(1i*phi0)/sqrt(N);
loop_ind = 1;
fig = figure;
% start loop
while(1)
    % Forwards prop.
    T = fftshift(fft(fftshift(t0)))/sqrt(N);
    Amp = abs(T); Phi = angle(T); I = abs(Amp).^2;
    % show progress
    if mod(loop_ind,1)==0
        pause(0.1)
        set(groot,'CurrentFigure',fig);
        stem(1:N,I,'linewidth',2); hold on
        stem(orders, I(index),'linewidth',2); hold off
        title(['Loop # ' num2str(loop_ind)])
        xlabel('Orders');ylabel('Eff.');
        grid on
        grid minor
        set(gca, 'linewidth',2, 'fontsize', 18);
        drawnow
    end
    % set target Amp.
    T(index) = exp(1i*Phi(index))/sqrt(length(orders));
    T(~index) = 0;
    % Backwards prop.
    t = ifftshift(ifft(ifftshift(T)))*sqrt(N);
    Phi = angle(t);
    % set source Amp.
    t = exp(1i*Phi)/sqrt(N);
    % check stop criteria
    if sqrt(mean(abs(t-t0).^2))/mean(abs(t0))<0.0001
        disp('Reach stop criteria')
        break
    elseif loop_ind>500
        disp('Reach Max. Loop #')
        break
    end
    % update
    t0 = t;
    loop_ind = loop_ind+1;
end
phi = angle(t);
end