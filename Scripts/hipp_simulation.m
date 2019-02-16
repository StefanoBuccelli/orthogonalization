clear
clc
close all
%% Trying to replicate supplementary figures from hipp (simulation) 

% we generated complex random numbers with Rayleigh distributed amplitude and random phase
% As a model of frequency transformed physiological
% signals, we generated complex random numbers with Rayleigh distributed amplitude and
% random phase. This corresponds to the frequency transform of Gaussian noise in the timedomain. Although this signal model is very general, the conclusions drawn from the present
% simulations are limited to this model. 

% param

signal_len=1e4;
for B=2 % Rayleigh parameter (equal to the std of the two original gauss distributions)

%%
% x = raylrnd(B,1,signal_len) +1i*raylrnd(B,1,signal_len);
% x_2 = raylrnd(B+1,1,signal_len) +1i*raylrnd(B+1,1,signal_len);
rng(4)
ampiezza=raylrnd(B,1,signal_len);
rng(4)
p = raylcdf(ampiezza,B);
ampiezza_remapped=1-p;
rng(4)
ampiezza_back_trasf=raylinv(ampiezza_remapped,B);
phases_random=rand(1,signal_len).*2*pi;
phases_random=phases_random-mean(phases_random);
phases_random_2=rand(1,signal_len).*2*pi;
phases_random_2=phases_random_2-mean(phases_random_2);
phases_random_3=rand(1,signal_len).*2*pi;
phases_random_3=phases_random_3-mean(phases_random_3);

x = ampiezza.*exp(1i.*phases_random);
x_2 = ampiezza_back_trasf.*exp(1i.*phases_random_2);
x_3=raylrnd(B,1,signal_len).*exp(1i.*phases_random_3);
% x = raylrnd(B,1,signal_len).*exp(1i.*raylrnd(B,1,signal_len));
% x_2 = raylrnd(B,1,signal_len).*exp(1i.*raylrnd(B,1,signal_len));

%% histogram of x and x_2 amplitudes and phases
figure
h_x(1)=subplot(2,3,1);
histogram(abs(x))
title('x amplitude')
h_x(1)=subplot(2,3,2);
histogram(angle(x))
title('x phase')
h_x(1)=subplot(2,3,3);
histogram(real(x))
title('x real')
h_x(1)=subplot(2,3,4);
histogram(abs(x_2))
title('x2 amplitude')
h_x(1)=subplot(2,3,5);
histogram(angle(x_2))
title('x2 phase')
h_x(1)=subplot(2,3,6);
histogram(real(x_2))
title('x2 real')
%%
range_coherence=-1:.1:1;

%% initialize variables
y=zeros(length(range_coherence),signal_len);
x_power=zeros(length(range_coherence),signal_len);
x_2_power=zeros(length(range_coherence),signal_len);
y_power=zeros(length(range_coherence),signal_len);
y_ort_x=zeros(length(range_coherence),signal_len);
x_ort_y=zeros(length(range_coherence),signal_len);
y_ort_x_power=zeros(length(range_coherence),signal_len);
x_ort_y_power=zeros(length(range_coherence),signal_len);

rho_all_plain=zeros(length(range_coherence),1);
rho_all_ort=zeros(length(range_coherence),1);
rho_all_x_x_ort_y=zeros(length(range_coherence),1);
rho_all_x_y_ort_x=zeros(length(range_coherence),1);
rho_all_y_x_ort_y=zeros(length(range_coherence),1);
rho_all_y_y_ort_x=zeros(length(range_coherence),1);
rho_all_x2_x_ort_y=zeros(length(range_coherence),1);
rho_all_x2_y_ort_x=zeros(length(range_coherence),1);

%% cycle over different values of choerence
curr_step=1;
for curr_c=range_coherence
    c = curr_c;
    % signals
    %      y(curr_step,:) = (c.*x) + (sqrt((1 - c^2)).*x_2);
    if c<=0
        y(curr_step,:) = ((c).*x_2) + ((sqrt(1-c^2)).*x_3);
    else
        y(curr_step,:) = (c.*x) + (sqrt(1-c^2).*x_3);% @c=-1 y=x_2(anticorr with x) @ c=0 y=un segnale che non centra un cazzo
        % @c=1 y=x(anticorr with x_2)
    end
%     y(curr_step,:) = (c.*real(x)+1i.*imag(x)) + (sqrt((1 - c^2)).*real(x_2)+1i.*(imag(x_2)));
    % power
    x_power(curr_step,:)=abs(x).^2;
    x_2_power(curr_step,:)=abs(x_2).^2;
    y_power(curr_step,:)=abs(y(curr_step,:)).^2;
    
    x_power(curr_step,:)=log10(x_power(curr_step,:));
    x_2_power(curr_step,:)=log10(x_2_power(curr_step,:));
    y_power(curr_step,:)=log10(y_power(curr_step,:));
    %% plain
    rho_plain=corrcoef(x_power(curr_step,:),y_power(curr_step,:));
    rho_all_plain(curr_step)=rho_plain(1,2);

    %% orthogonalization
    y_ort_x(curr_step,:)=y(curr_step,:)-real(x.*conj(y(curr_step,:))./(abs(x).^2)).*x; %from hipp
    x_ort_y(curr_step,:)=x-real(y(curr_step,:).*conj(x)./(abs(y(curr_step,:)).^2)).*y(curr_step,:); %from hipp
    
    y_ort_x_power(curr_step,:)=abs(y_ort_x(curr_step,:)).^2;
    x_ort_y_power(curr_step,:)=abs(x_ort_y(curr_step,:)).^2;
    
    y_ort_x_power(curr_step,:)=log10(y_ort_x_power(curr_step,:));
    x_ort_y_power(curr_step,:)=log10(x_ort_y_power(curr_step,:));
    %% orthog corr ortog
    rho_ort=corrcoef(x_ort_y_power(curr_step,:),y_ort_x_power(curr_step,:));
    rho_all_ort(curr_step)=rho_ort(1,2);
    
    %% x corr x ortog y
    rho_x_x_ort_y=corrcoef(x_power(curr_step,:),x_ort_y_power(curr_step,:));
    rho_all_x_x_ort_y(curr_step)=rho_x_x_ort_y(1,2);
   
    %% x corr y ortog x
    rho_x_y_ort_x=corrcoef(x_power(curr_step,:),y_ort_x_power(curr_step,:));
    rho_all_x_y_ort_x(curr_step)=rho_x_y_ort_x(1,2);
    
    %% y corr x ortog y
    rho_y_x_ort_y=corrcoef(y_power(curr_step,:),x_ort_y_power(curr_step,:));
    rho_all_y_x_ort_y(curr_step)=rho_y_x_ort_y(1,2);
    
    %% y corr y ortog x
    rho_y_y_ort_x=corrcoef(y_power(curr_step,:),y_ort_x_power(curr_step,:));
    rho_all_y_y_ort_x(curr_step)=rho_y_y_ort_x(1,2);
   
    %% x_2 corr x ortog y
    rho_x2_x_ort_y=corrcoef(x_2_power(curr_step,:),x_ort_y_power(curr_step,:));
    rho_all_x2_x_ort_y(curr_step)=rho_x2_x_ort_y(1,2);
    
    %% x_2 corr y ortog x
    rho_x2_y_ort_x=corrcoef(x_2_power(curr_step,:),y_ort_x_power(curr_step,:));
    rho_all_x2_y_ort_x(curr_step)=rho_x2_y_ort_x(1,2);

    curr_step=curr_step+1;
end
%%
ort_hipp=(atanh(rho_all_x_y_ort_x)./2+atanh(rho_all_y_x_ort_y)./2);
ort_hipp_no_atan=((rho_all_x_y_ort_x)./2+(rho_all_y_x_ort_y)./2);
figure
subplot(2,1,1)
plot(range_coherence,ort_hipp)
hold on
plot(range_coherence,ort_hipp_no_atan)
hold on
plot(range_coherence,rho_all_plain,'k')
title(['rho (ort hipp) vs rho plain ' num2str(B)])
subplot(2,1,2)
plot(rho_all_plain,ort_hipp_no_atan,'k')
m=mean(diff(ort_hipp_no_atan(1:7))./diff(rho_all_plain(1:7)))
title('rho all plain - rho ort hipp')
end

%% figure to compare the different correlations
figure
h_s(1)=subplot(1,4,1);
plot(range_coherence,rho_all_plain,'k.')
hold on
plot(range_coherence,rho_all_ort,'m.')
legend({'plain','ort ort'},'interpreter','none')
xlabel('coherence')
ylabel('correlation coeff')

h_s(2)=subplot(1,4,2);
plot(range_coherence,rho_all_plain,'k.')
hold on
plot(range_coherence,rho_all_x_x_ort_y,'bo')
plot(range_coherence,rho_all_x_y_ort_x,'b.')
legend({'plain','x_x_ort_y','x_y_ort_x'},'interpreter','none')
xlabel('coherence')
ylabel('correlation coeff')

h_s(3)=subplot(1,4,3);
plot(range_coherence,rho_all_plain,'k.')
hold on
plot(range_coherence,rho_all_x2_x_ort_y,'go')
plot(range_coherence,rho_all_x2_y_ort_x,'g.')
legend({'plain','x2_x_ort_y','x2_y_ort_x'},'interpreter','none')
xlabel('coherence')
ylabel('correlation coeff')

h_s(4)=subplot(1,4,4);
plot(range_coherence,rho_all_plain,'k.')
hold on
plot(range_coherence,rho_all_y_x_ort_y,'ro')
plot(range_coherence,rho_all_y_y_ort_x,'r.')
legend({'plain','y_x_ort_y','y_y_ort_x'},'interpreter','none')
xlabel('coherence')
ylabel('correlation coeff')

linkaxes(h_s,'xy')
xlim([-1.1 1.1])
ylim([-1.1 1.1])
xlabel('coherence')
ylabel('correlation coeff')

%% comparing phases
x_phases=angle(x).*180/pi;
x_2_phases=angle(x_2).*180/pi;
y_phases=angle(y).*180/pi;
y_ort_x_phases=angle(y_ort_x).*180/pi;
x_ort_y_phases=angle(x_ort_y).*180/pi; 

figure
subplot(1,3,1)
imagesc(abs(x_phases-x_2_phases))
title('diff abs(x - x2) phases')
subplot(1,3,2)
imagesc(abs(x_phases-y_ort_x_phases))
title('diff abs(x - y ort x) phases')
subplot(1,3,3)
imagesc(abs(y_phases-x_ort_y_phases))
title('diff abs(y - x ort y) phases')

%% check different phases
curr_step=1; % curr step for coherence level to show
figure
h_p(1)=subplot(1,3,1);
histogram(x_phases*180/pi,'normalization','probability')
xlabel('x phase distribution [deg]')
ylabel('probability')
title('x phase histogram [deg]')
h_p(2)=subplot(1,3,2);
histogram(x_2_phases*180/pi,'normalization','probability')
xlabel('x phase distribution [deg]')
ylabel('probability')
title('x2 phase histogram [deg]')
h_p(3)=subplot(1,3,3);
histogram(y_phases(curr_step,:)*180/pi,'normalization','probability')
xlabel('x phase distribution [deg]')
ylabel('probability')
title(['y (' num2str(curr_step) ') phase histogram [deg]'])
linkaxes(h_p,'xy')
%% comparing signal abs at specific levels of coherence
curr_step=1;
figure
h(1)=subplot(2,1,1);
plot(sqrt(x_power(curr_step,:)),'m')
hold on
plot(real(x),'b')
legend({'abs(x)','real(x)'},'interpreter','none')
title('comparing x signal and abs')
h(2)=subplot(2,1,2);
plot(sqrt(y_power(curr_step,:)),'m')
hold on
plot(real(y(curr_step,:)),'r')
legend({'abs(y)','real(y)'},'interpreter','none')
title(['comparing y signal and abs in curr step = ' num2str(curr_step)])
linkaxes(h,'x')


%% compass plot
figure
for curr_sample=1:10
    subplot(2,5,curr_sample)
    for curr_cohere_inx=1:1:length(range_coherence)
        compass(real(y(curr_cohere_inx,curr_sample)),imag(y(curr_cohere_inx,curr_sample)),'r')       
        hold on
        compass(real(y_ort_x(curr_cohere_inx,curr_sample)),imag(y_ort_x(curr_cohere_inx,curr_sample)),'k')
        compass(real(x_ort_y(curr_cohere_inx,curr_sample)),imag(x_ort_y(curr_cohere_inx,curr_sample)),'m')
    end
    compass(real(x(curr_sample)),imag(x(curr_sample)),'b')
    compass(real(x_2(curr_sample)),imag(x_2(curr_sample)),'g')
    title(num2str(curr_sample))
end