function U = spherical_wave(F, N, pitch, lam)
    % spherical_wave generates the complex amplitude of a 1D spherical wave
    % F - focal length (distance between point source and sampling plane)
    % N - number of points
    % pitch - sampling interval
    % lam - wavelength
    if isinf(F)
        U = ones(1,N);
        return
    end
    % Define the spatial coordinates
    x = (-N/2:N/2-1) * pitch; % N-point spatial coordinates centered around zero
    
    % Calculate the distance from each sampling point to the point source
    r = sqrt(x.^2 + F^2);
    
    % Calculate the wavenumber
    k = 2 * pi / lam;
    
    % Calculate the complex amplitude of the spherical wave
    U = exp(1i * k * r);    
end