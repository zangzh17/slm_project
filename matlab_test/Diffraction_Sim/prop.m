function T = prop(t,H,pad_val,init_wavefront)
% calculate Angular Spectrum Method (ASM) or Rayleigh-Sommerfield Convolution (RSC)
% parameters
t = t(:).';
if nargin<3
    pad_val = 0;
end
% padding: "circular" or 0
pad_x = round((length(H)-length(t))/2);
t_pad = padarray(t,[0,pad_x],pad_val);

% add initial wavefront
if nargin==4
    t_pad = t_pad.*init_wavefront;
end

% calc ASM/RSC
t_pad_FT = fftshift(fft(fftshift(t_pad)));
t_pro = ifftshift(ifft(ifftshift(t_pad_FT.*H,2),[],2),2);
T = t_pro(:, pad_x+1:pad_x+length(t));
end
