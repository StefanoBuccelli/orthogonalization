%% understandong rayleight

%% using the definition of Reyleigh from wikipedia and comparing with matlab
gauss_variance=2;
bin_width=.1;
gauss_1=sqrt(gauss_variance)*randn(1,signal_len);
gauss_2=sqrt(gauss_variance)*randn(1,signal_len);
reyl_from_2_gauss=sqrt(gauss_1.^2 + gauss_2.^2);
figure
histogram(gauss_1,'binwidth',bin_width)
hold on
histogram(gauss_2,'binwidth',bin_width)
histogram(reyl_from_2_gauss,'binwidth',bin_width)
histogram(raylrnd(sqrt(gauss_variance),1,signal_len),'binwidth',bin_width) % the parameter for matlab is not the variance but the std
legend({'gauss 1','gauss 2','reyl from gauss','reyl from matlab'})
title('comparing distributions')