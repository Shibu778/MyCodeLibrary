% Testing the rosenstein algorithm to calculate the largest lyapunov exponent
%x = rand(500,1);
x = xlsread('hennon_x_dt_20_01.xlsx', 'Sheet1', 'A1:A500'); % Read your data using this file
m = 5;	% Embedding dimension
sample = 1; % Sampling time
tao = 13;      % Time delay
N = length(x); % Length of the data
maxiter = 30; % Maximum tranjectory tracking to be considered
meanperiod = 60;	% Theiler window or the meanperiod that should exist between two neighbouring points
d = lyarosenstein(x, m, tao, meanperiod, maxiter);		% Calculates the average logrithm of divergence with different
														% tracking time (Please see the lyarosenstein.m file for details about inputs)
tlinear=1:maxiter;	% Generate a sequence representing tracking length
f = polyfit(sample*(tlinear),d(tlinear),1);	% Fitting the sequence of average divergence with the time in second
f % Prints fitted parameter, f(1) is the slope and f(2) is the intercept
printf("The Lyapunov exponent is ");
disp(f(2)*sample);

% Plotting the divergence plot
figure
title('Divergence plot')
plot(sample*tlinear,d, "linewidth", 4)

% Plotting the divergence plot along with the fitted line
figure
title('Maximum Lyapunov exponent for the Lorenz system')
plot(sample*tlinear,d, "linewidth", 4)
hold on;
%plot(sample * (tlinear), sample * (tlinear) * 1.50,'-', "linewidth", 4 )
plot(sample*(tlinear), sample*(tlinear)*f(1)+f(2), '.-', "linewidth", 4)
legend('Calculated d vs t', 'Fitted d vs t fit')
xlabel('Time t')
ylabel('Average divergence')
set(gca, "linewidth", 4, "fontsize", 12)
hold off;


