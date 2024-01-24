% Testing the rosenstein algorithm to calculate the largest lyapunov exponent
x = rand(500,1);
m = 3;
tao = 10;
maxiter = 500;
meanperiod = 500;
d = lyarosenstein(x, m, tao, meanperiod, maxiter);

figure
plot(d)
%% LLE Calculation
fs=2000;%sampling frequency
tlinear=15:78;
F = polyfit(tlinear,d(tlinear),1);
lle = F(1)*fs