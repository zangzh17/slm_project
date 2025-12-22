function zc = critical_range(pitch,lam,Nx)
zc = 2*Nx*pitch^2/lam.*sqrt(1-(lam./(2*pitch))^2);