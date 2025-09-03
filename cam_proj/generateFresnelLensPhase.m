function phase_pattern_uint8 = generateFresnelLensPhase(width, height, focal_length_mm, angle_x_mrad, angle_y_mrad, phase_scale, wavelength, pixel_size)
    % Direct 8-bit computation with LUTs
    if nargin < 5, phase_scale = 1.0; end
    if nargin < 6, wavelength = 550e-9; end
    if nargin < 7, pixel_size = 9.2e-6; end
    
    % Constants
    focal_length_m = focal_length_mm * 1e-3;
    k = 2 * pi / wavelength;
    f_squared = focal_length_m^2;
    
    % Use single precision for coordinates (still need some precision here)
    [x, y] = meshgrid(single(1:width), single(1:height));
    
    center_x = (width + 1) * 0.5;
    center_y = (height + 1) * 0.5;
    
    x_meters = (x - center_x) * pixel_size;
    y_meters = (y - center_y) * pixel_size;
    
    % Calculate phase and directly quantize to 8-bit
    r_squared = x_meters.^2 + y_meters.^2;
    lens_phase = k * (focal_length_m - sqrt(f_squared + r_squared));
    
    % Add steering if needed
    if angle_x_mrad ~= 0 || angle_y_mrad ~= 0
        phase_gradient_x = k * sin(angle_x_mrad * 1e-3);
        phase_gradient_y = k * sin(angle_y_mrad * 1e-3);
        lens_phase = lens_phase + phase_gradient_x * x_meters + phase_gradient_y * y_meters;
    end
    
    % Direct conversion to 8-bit (0-255) with modulo built-in
    scale_factor = 255 / (2*pi) * phase_scale;
    phase_pattern_uint8 = uint8(mod(lens_phase * scale_factor, 256));
end