function phi = IFTA1D_ASM(I_obj,z,pitch,lam)
% opt Diffraction pattern I_obj
% z,pitch,lam: mm
N = length(I_obj);
I_obj = I_obj/max(I_obj);
phi0 = 2*pi*rand(1,N);
loop_ind = 1;
fig = figure;
% start loop
while(1)
    % set source Amp.
    t = exp(1i*phi0);
    % Forwards prop.
    T = ASM1D(t,z,pitch,lam);
    phi = angle(T);
    % show progress
    if mod(loop_ind,1)==0
        set(groot,'CurrentFigure',fig);
        plot(abs(T).^2,'linewidth',1);
        title(['Loop # ' num2str(loop_ind)])
        xlabel('Orders');ylabel('Eff.');
        grid on
        grid minor
        set(gca, 'linewidth',1.5, 'fontsize', 16);
        drawnow
    end
    % set target Amp.
    T = exp(1i*phi).*sqrt(I_obj);
    % Backwards prop.
    t = ASM1D(T,-z,pitch,lam);
    phi = angle(t);
    % check stop criteria
    if sqrt(mean(abs(phi-phi0).^2))<0.0001
        disp('Reach stop criteria')
        break
    elseif loop_ind>500
        disp('Reach Max. Loop #')
        break
    end
    % update
    phi0 = phi;
    loop_ind = loop_ind+1;
end
end