%% understanding orthogonalization
clear
clc
close all
% param
rng(1)
signal_len=1e4;
B=1; % Rayleigh parameter
% two complex signals (synthetic spectra)
x = rand(1,signal_len) +1i*rand(1,signal_len);
y = rand(1,signal_len) +1i*rand(1,signal_len);

y_ort_x=y(1)-real(x(1)*conj(y(1))/(abs(x(1))^2))*x(1); % from hipp
% y_ort_x_versor=imag(y(1)*conj(x)/abs(x(1)))*1i*x(1)/abs(x(1)); % equivalent with versor.. same number
x_ort_y=x(1)-real(y(1)*conj(x(1))/(abs(y(1))^2))*y(1); % from hipp

phase_difference=(angle(x(1))-angle(y_ort_x(1)))*180/pi; % should be 90 deg

%% ortogonalization 
figure
compass(real(x(1)),imag(x(1)),'b')
hold on
compass(real(y(1)),imag(y(1)),'r')
compass(real(y_ort_x(1)),imag(y_ort_x(1)),'r-o')
compass(real(x_ort_y(1)),imag(x_ort_y(1)),'b-o')
legend({'x','y','y ort x','x ort y'})
title('showing 2 samples with orthogonalization')
