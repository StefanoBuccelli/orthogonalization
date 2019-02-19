%% Trying to replicate supplementary figures from hipp 2012 (simulation)
clear
clc
close all

% Change the current folder to the folder of this m-file.
if(~isdeployed)
  cd(fileparts(which(mfilename)));
end
cd ..
cd('Figures')
%% signal parameters
signal_len=1e4; % samples
B = 2; % Rayleigh parameter (equal to the std of the two original gauss distributions)
%% creating signals
rng(4)
amplitude=raylrnd(B,1,signal_len);
p = raylcdf(amplitude,B);
amplitude_remapped=1-p;
amplitude_back_transf=raylinv(amplitude_remapped,B);

phases_random=rand(1,signal_len).*2*pi;
phases_random=phases_random-mean(phases_random);
phases_random_2=rand(1,signal_len).*2*pi;
phases_random_2=phases_random_2-mean(phases_random_2);
phases_random_3=rand(1,signal_len).*2*pi;
phases_random_3=phases_random_3-mean(phases_random_3);
for curr_mult=1:7
x = amplitude.*exp(1i.*phases_random);
x_2 = amplitude.*exp(1i.*(phases_random+curr_mult*pi/16));
% x_2 = amplitude_back_transf.*exp(1i.*phases_random_2);
x_3=raylrnd(B,1,signal_len).*exp(1i.*phases_random_3);

%% showing signal properties histogram of x and x_2 amplitudes and phases
% figure
% h_x(1)=subplot(3,4,1);
% histogram(real(x))
% title('real(x)','interpreter','none')
% h_x(2)=subplot(3,4,2);
% histogram(imag(x))
% title('imag(x)','interpreter','none')
% h_x(3)=subplot(3,4,3);
% histogram(abs(x))
% title('abs(x)','interpreter','none')
% h_x(4)=subplot(3,4,4);
% histogram(angle(x)*180/pi)
% title('phase(x)','interpreter','none')
% 
% h_x(1)=subplot(3,4,5);
% histogram(real(x_2))
% title('real(x_2)','interpreter','none')
% h_x(2)=subplot(3,4,6);
% histogram(imag(x_2))
% title('imag(x_2)')
% h_x(3)=subplot(3,4,7);
% histogram(abs(x_2))
% title('abs(x_2)','interpreter','none')
% h_x(4)=subplot(3,4,8);
% histogram(angle(x_2)*180/pi)
% title('phase(x_2)','interpreter','none')
% 
% h_x(1)=subplot(3,4,9);
% histogram(real(x_3))
% title('real(x_3)','interpreter','none')
% h_x(2)=subplot(3,4,10);
% histogram(imag(x_3))
% title('imag(x_3)','interpreter','none')
% h_x(3)=subplot(3,4,11);
% histogram(abs(x_3))
% title('abs(x_3)','interpreter','none')
% h_x(4)=subplot(3,4,12);
% histogram(angle(x_3)*180/pi)
% title('phase(x_3)','interpreter','none')
% 
% linkaxes(h_x,'y')
% savefig(gcf,['signals B = ' num2str(B)])
% saveas(gcf,['signals B = ' num2str(B)],'tif')

%% initialize variables
% range_coherence=-1:.01:1;
% curr_step=1;
range_coherence=0.1:0.01:1;
curr_step=1;
y=zeros(length(range_coherence),signal_len);
x_power=zeros(length(range_coherence),signal_len);
x_2_power=zeros(length(range_coherence),signal_len);
x_3_power=zeros(length(range_coherence),signal_len);
y_power=zeros(length(range_coherence),signal_len);
y_ort_x=zeros(length(range_coherence),signal_len);
x_ort_y=zeros(length(range_coherence),signal_len);
y_ort_x_power=zeros(length(range_coherence),signal_len);
x_ort_y_power=zeros(length(range_coherence),signal_len);

rho_all_x_y_plain=zeros(length(range_coherence),1);
rho_all_ort=zeros(length(range_coherence),1);
rho_all_x_x_ort_y=zeros(length(range_coherence),1);
rho_all_x_y_ort_x=zeros(length(range_coherence),1);
rho_all_y_x_ort_y=zeros(length(range_coherence),1);
rho_all_y_y_ort_x=zeros(length(range_coherence),1);
rho_all_x2_x_ort_y=zeros(length(range_coherence),1);
rho_all_x2_y_ort_x=zeros(length(range_coherence),1);

%% cycle over different values of choerence

for curr_c=range_coherence
    c = curr_c;
    % signals
    %      y(curr_step,:) = (c.*x) + (sqrt((1 - c^2)).*x_2);
    if c<=0
        y(curr_step,:) = ((c).*x_2) + (sqrt(1-c^2).*x_3);% @c=-1 y=x_2(anticorr with x) @ c=0 y=an uncorrelated signal
    else
        y(curr_step,:) = (c.*x_2) + (sqrt(1-c^2).*x_3); % @c=1 y=x(anticorr with x_2)
    end
    % power
    x_power(curr_step,:)=abs(x).^2;
    x_2_power(curr_step,:)=abs(x_2).^2;
    x_3_power(curr_step,:)=abs(x_3).^2;
    y_power(curr_step,:)=abs(y(curr_step,:)).^2;
    
    x_power(curr_step,:)=log10(x_power(curr_step,:));
    x_2_power(curr_step,:)=log10(x_2_power(curr_step,:));
    x_3_power(curr_step,:)=log10(x_3_power(curr_step,:));
    y_power(curr_step,:)=log10(y_power(curr_step,:));
    
    %% figure to compare power signals
%     figure
%     plot(x_power(curr_step,:))
%     hold on
%     plot(x_2_power(curr_step,:))
%     plot(x_3_power(curr_step,:))
%     legend({'x','x_2','x_3'},'interpreter','none')
%     title('comparing power envelopes')
%     xlabel('samples')
%     ylabel('log10( |signal|^2 )')    
    
    %% plain
    rho_all_x_y_plain(curr_step)=corr(x_power(curr_step,:)',y_power(curr_step,:)');
    rho_all_x_x2_plain(curr_step)=corr(x_power(curr_step,:)',x_2_power(curr_step,:)');
    rho_all_x_x3_plain(curr_step)=corr(x_power(curr_step,:)',x_3_power(curr_step,:)');
    rho_all_x2_x3_plain(curr_step)=corr(x_2_power(curr_step,:)',x_3_power(curr_step,:)');
    disp(['correlation between x-x_2: ' num2str(rho_all_x_x2_plain(curr_step)) ' , x-x_3: ' num2str(rho_all_x_x3_plain(curr_step)) ' , x_2-x_3: ' num2str(rho_all_x2_x3_plain(curr_step))])
    
    %% orthogonalization x-y
    y_ort_x(curr_step,:)=y(curr_step,:)-real(x.*conj(y(curr_step,:))./(abs(x).^2)).*x; %from hipp equivalent when powered : siems=imag(y(curr_step,:).*conj(x)./abs(x));
    x_ort_y(curr_step,:)=x-real(y(curr_step,:).*conj(x)./(abs(y(curr_step,:)).^2)).*y(curr_step,:); %from hipp equivalent when powered : siems=imag(x.*conj(y(curr_step,:))./abs(y(curr_step,:)));
    
    y_ort_x_power(curr_step,:)=abs(y_ort_x(curr_step,:)).^2;
    x_ort_y_power(curr_step,:)=abs(x_ort_y(curr_step,:)).^2;
%     
    y_ort_x_power(curr_step,:)=log10(y_ort_x_power(curr_step,:));
    x_ort_y_power(curr_step,:)=log10(x_ort_y_power(curr_step,:));
    
    %% orthogonalization x2-y
    y_ort_x2(curr_step,:)=y(curr_step,:)-real(x_2.*conj(y(curr_step,:))./(abs(x_2).^2)).*x_2; %from hipp
    x2_ort_y(curr_step,:)=x_2-real(y(curr_step,:).*conj(x_2)./(abs(y(curr_step,:)).^2)).*y(curr_step,:); %from hipp
    x_ort_x2(curr_step,:)=x-real(x_2.*conj(x)./(abs(x_2).^2)).*x_2; %from hipp
    x2_ort_x(curr_step,:)=x_2-real(x.*conj(x_2)./(abs(x).^2)).*x; %from hipp
    
    y_ort_x2_power(curr_step,:)=abs(y_ort_x2(curr_step,:)).^2;
    x2_ort_y_power(curr_step,:)=abs(x2_ort_y(curr_step,:)).^2;
    x_ort_x2_power(curr_step,:)=abs(x_ort_x2(curr_step,:)).^2;
    x2_ort_x_power(curr_step,:)=abs(x2_ort_x(curr_step,:)).^2;
%     
    y_ort_x2_power(curr_step,:)=log10(y_ort_x2_power(curr_step,:));
    x2_ort_y_power(curr_step,:)=log10(x2_ort_y_power(curr_step,:));
    x_ort_x2_power(curr_step,:)=log10(x_ort_x2_power(curr_step,:));
    x2_ort_x_power(curr_step,:)=log10(x2_ort_x_power(curr_step,:));
    
    %% x2 , y ort x2
    rho_all_x2_y_ort_x2(curr_step)=corr(y_ort_x2_power(curr_step,:)',x_2_power(curr_step,:)');
    %% x2 , x2 ort y
    rho_all_x2_x2_ort_y(curr_step)=corr(x2_ort_y_power(curr_step,:)',x_2_power(curr_step,:)');
    %% y , x2 ort y
    rho_all_y_x2_ort_y(curr_step)=corr(x2_ort_y_power(curr_step,:)',y_power(curr_step,:)');
    %% y , y ort x2
    rho_all_y_y_ort_x2(curr_step)=corr(y_ort_x2_power(curr_step,:)',y_power(curr_step,:)');
    %% x , x2 ort y
    rho_all_x_x2_ort_y(curr_step)=corr(x2_ort_y_power(curr_step,:)',x_power(curr_step,:)');
    %% x , y ort x2
    rho_all_x_y_ort_x2(curr_step)=corr(y_ort_x2_power(curr_step,:)',x_power(curr_step,:)');
    
    %% y , x ort x2
    rho_all_y_x_ort_x2(curr_step)=corr(x_ort_x2_power(curr_step,:)',y_power(curr_step,:)');
    %% y , x2 ort x
    rho_all_y_x2_ort_x(curr_step)=corr(x2_ort_x_power(curr_step,:)',y_power(curr_step,:)');
    
    %% orthog corr ortog
    rho_all_ort(curr_step)=corr(x_ort_y_power(curr_step,:)',y_ort_x_power(curr_step,:)');
    %% x corr x ortog y
    rho_all_x_x_ort_y(curr_step)=corr(x_power(curr_step,:)',x_ort_y_power(curr_step,:)');
    %% x corr y ortog x
    rho_all_x_y_ort_x(curr_step)=corr(x_power(curr_step,:)',y_ort_x_power(curr_step,:)');
    %% y corr x ortog y
    rho_all_y_x_ort_y(curr_step)=corr(y_power(curr_step,:)',x_ort_y_power(curr_step,:)');
    %% y corr y ortog x
    rho_all_y_y_ort_x(curr_step)=corr(y_power(curr_step,:)',y_ort_x_power(curr_step,:)');
    %% x_2 corr x ortog y
    rho_all_x2_x_ort_y(curr_step)=corr(x_2_power(curr_step,:)',x_ort_y_power(curr_step,:)');
    %% x_2 corr y ortog x
    rho_all_x2_y_ort_x(curr_step)=corr(x_2_power(curr_step,:)',y_ort_x_power(curr_step,:)');
    
    curr_step=curr_step+1;
end

figure(101)
plot(rho_all_x_y_plain,rho_all_x_y_ort_x)
hold on
xlabel('Correlation')
ylabel('Correlation orth')
end

%% figure to compare the different correlations putting both coherence and correlation as x
figure
h_s(1)=subplot(2,4,1);
plot(range_coherence,rho_all_x_y_plain,'k.')
hold on
plot(range_coherence,rho_all_ort,'m.')
legend({'plain','(x ort y , y ort x)'},'interpreter','none')
xlabel('coherence')
ylabel('correlation coeff')

h_s(2)=subplot(2,4,2);
plot(range_coherence,rho_all_x_y_plain,'k.')
hold on
plot(range_coherence,rho_all_x_x_ort_y,'bo')
plot(range_coherence,rho_all_x_y_ort_x,'b.')
legend({'plain','(x , x_ort_y)','(x , y_ort_x)'},'interpreter','none')
xlabel('coherence')
ylabel('correlation coeff')

h_s(3)=subplot(2,4,3);
plot(range_coherence,rho_all_x_y_plain,'k.')
hold on
plot(range_coherence,rho_all_x2_x_ort_y,'go')
plot(range_coherence,rho_all_x2_y_ort_x,'g.')
legend({'plain','(x2 , x_ort_y)','(x2 , y_ort_x)'},'interpreter','none')
xlabel('coherence')
ylabel('correlation coeff')

h_s(4)=subplot(2,4,4);
plot(range_coherence,rho_all_x_y_plain,'k.')
hold on
plot(range_coherence,rho_all_y_x_ort_y,'ro')
plot(range_coherence,rho_all_y_y_ort_x,'r.')
legend({'plain','(y , x_ort_y)','(y , y_ort_x)'},'interpreter','none')
xlabel('coherence')
ylabel('correlation coeff')

h_s(5)=subplot(2,4,5);
plot(rho_all_x_y_plain,rho_all_x_y_plain,'k.')
hold on
plot(rho_all_x_y_plain,rho_all_ort,'m.')
legend({'plain','(x ort y , y ort x)'},'interpreter','none')
xlabel('correlation')
ylabel('correlation coeff')

h_s(6)=subplot(2,4,6);
plot(rho_all_x_y_plain,rho_all_x_y_plain,'k.')
hold on
plot(rho_all_x_y_plain,rho_all_x_x_ort_y,'bo')
plot(rho_all_x_y_plain,rho_all_x_y_ort_x,'b.')
legend({'plain','(x , x_ort_y)','(x_, y_ort_x)'},'interpreter','none')
xlabel('correlation')
ylabel('correlation coeff')

h_s(7)=subplot(2,4,7);
plot(rho_all_x_y_plain,rho_all_x_y_plain,'k.')
hold on
plot(rho_all_x_y_plain,rho_all_x2_x_ort_y,'go')
plot(rho_all_x_y_plain,rho_all_x2_y_ort_x,'g.')
legend({'plain','(x2 , x_ort_y)','(x2 , y_ort_x)'},'interpreter','none')
xlabel('correlation')
ylabel('correlation coeff')

h_s(8)=subplot(2,4,8);
plot(rho_all_x_y_plain,rho_all_x_y_plain,'k.')
hold on
plot(rho_all_x_y_plain,rho_all_y_x_ort_y,'ro')
plot(rho_all_x_y_plain,rho_all_y_y_ort_x,'r.')
legend({'plain','(y , x_ort_y)','(y , y_ort_x)'},'interpreter','none')
xlabel('correlation')
ylabel('correlation_coherence_corr')

linkaxes(h_s,'xy')
xlim([-1.1 1.1])
ylim([-1.1 1.1])
savefig(gcf,['correlations B = ' num2str(B)])
saveas(gcf,['correlations B = ' num2str(B)],'tif')

%% showing phase differences
x_phases_deg=angle(x).*180/pi;
x_2_phases_deg=angle(x_2).*180/pi;
y_phases_deg=angle(y).*180/pi;
y_ort_x_phases_deg=angle(y_ort_x).*180/pi;
x_ort_y_phases_deg=angle(x_ort_y).*180/pi;

figure
subplot(1,3,1)
imagesc(abs(x_phases_deg-x_2_phases_deg))
title('diff abs(x - x2) phases')
xlabel('samples')
subplot(1,3,2)
imagesc(abs(x_phases_deg-y_ort_x_phases_deg))
title('diff abs(x - y ort x) phases')
xlabel('samples')
ylabel('range of coherence -1 to 1')
subplot(1,3,3)
imagesc(abs(y_phases_deg-x_ort_y_phases_deg))
title('diff abs(y - x ort y) phases')
xlabel('samples')
ylabel('range of coherence -1 to 1')
savefig(gcf,'phase differences')
saveas(gcf,'phase differences','tif')
%% check different phases
% curr_step=1; % curr step for coherence level to show
% figure
% h_p(1)=subplot(1,3,1);
% histogram(x_phases_deg,'normalization','probability')
% xlabel('x phase distribution [deg]')
% ylabel('probability')
% title('x phase histogram [deg]')
% h_p(2)=subplot(1,3,2);
% histogram(x_2_phases_deg,'normalization','probability')
% xlabel('x phase distribution [deg]')
% ylabel('probability')
% title('x2 phase histogram [deg]')
% h_p(3)=subplot(1,3,3);
% histogram(y_phases_deg(curr_step,:),'normalization','probability')
% xlabel('x phase distribution [deg]')
% ylabel('probability')
% title(['y (' num2str(curr_step) ') phase histogram [deg]'])
% linkaxes(h_p,'xy')


%% comparing signal abs at specific levels of coherence
curr_step=21;
figure
h(1)=subplot(2,1,1);
plot(sqrt(10.^(x_power(curr_step,:))),'m')
hold on
plot(real(x),'b')
legend({'abs(x)','real(x)'},'interpreter','none')
title('comparing x signal and abs')
h(2)=subplot(2,1,2);
plot(sqrt(10.^real(y_power(curr_step,:))),'m')
hold on
plot(real(y(curr_step,:)),'r')
legend({'abs(y)','real(y)'},'interpreter','none')
title(['comparing y signal and abs in curr step = ' num2str(curr_step)])
linkaxes(h,'x')
savefig(gcf,['envelope_curr_step = ' num2str(curr_step)])
saveas(gcf,['envelope_curr_step = ' num2str(curr_step)],'tif')
%% compass plot @ 10 different samples
figure
samples=1:10;
h_compass=zeros(1,length(samples));
% reduce number of arrows 
if length(range_coherence)>10
    range_coherence_subplot=linspace(1,length(range_coherence),10);
else
    range_coherence_subplot=range_coherence;
end
%%
% for curr_sample=samples
%     %     h_compass(curr_sample)=subplot(2,5,curr_sample);
%     figure
%     for curr_cohere_inx=1:1:length(range_coherence_subplot)
%         compass(real(y(curr_cohere_inx,curr_sample)),imag(y(curr_cohere_inx,curr_sample)),'r')
%         hold on
%         compass(real(y_ort_x(curr_cohere_inx,curr_sample)),imag(y_ort_x(curr_cohere_inx,curr_sample)),'k')
%         compass(real(x_ort_y(curr_cohere_inx,curr_sample)),imag(x_ort_y(curr_cohere_inx,curr_sample)),'m')
%     end
%     compass(real(x(curr_sample)),imag(x(curr_sample)),'b')
%     compass(real(x_2(curr_sample)),imag(x_2(curr_sample)),'g')
%     compass(real(x_3(curr_sample)),imag(x_3(curr_sample)),'y')
%     title(num2str(curr_sample))
%     savefig(gcf,['compass_sample_' num2str(curr_sample)])
%     saveas(gcf,['compass_sample_' num2str(curr_sample)],'tif')
% end
% linkaxes(h_compass,'xy')


%% comparing cross correlations for all possible pairs (more or less)
figure
h_s2(1)=subplot(2,4,[1 5]);
plot(rho_all_x_y_plain,rho_all_x_y_plain,'k.')
hold on
plot(rho_all_x_y_plain,rho_all_ort,'m.')
legend({'plain','(x ort y , y ort x)'},'interpreter','none')
xlabel('correlation')
ylabel('correlation coeff')

h_s2(2)=subplot(2,4,2);
plot(rho_all_x_y_plain,rho_all_x_y_plain,'k.')
hold on
plot(rho_all_x_y_plain,rho_all_x_x_ort_y,'bo')
plot(rho_all_x_y_plain,rho_all_x_y_ort_x,'b.')
legend({'plain','(x , x_ort_y)','(x_, y_ort_x)'},'interpreter','none')
xlabel('correlation')
ylabel('correlation coeff')

h_s2(3)=subplot(2,4,3);
plot(rho_all_x_y_plain,rho_all_x_y_plain,'k.')
hold on
plot(rho_all_x_y_plain,rho_all_x2_x_ort_y,'go')
plot(rho_all_x_y_plain,rho_all_x2_y_ort_x,'g.')
legend({'plain','(x2 , x_ort_y)','(x2 , y_ort_x)'},'interpreter','none')
xlabel('correlation')
ylabel('correlation coeff')

h_s2(4)=subplot(2,4,4);
plot(rho_all_x_y_plain,rho_all_x_y_plain,'k.')
hold on
plot(rho_all_x_y_plain,rho_all_y_x_ort_y,'ro')
plot(rho_all_x_y_plain,rho_all_y_y_ort_x,'r.')
legend({'plain','(y , x_ort_y)','(y , y_ort_x)'},'interpreter','none')
xlabel('correlation')
ylabel('correlation coeff')

h_s2(6)=subplot(2,4,6);
plot(rho_all_x_y_plain,rho_all_x_y_plain,'k.')
hold on
plot(rho_all_x_y_plain,rho_all_x_x2_ort_y,'bo')
plot(rho_all_x_y_plain,rho_all_x_y_ort_x2,'b.')
plot(rho_all_x_y_plain,rho_all_x_y_ort_x2,'b*')
legend({'plain','(x , x2 ort y)','(x , y ort x2)'},'interpreter','none')
xlabel('correlation')
ylabel('correlation coeff')

h_s2(7)=subplot(2,4,7);
plot(rho_all_x_y_plain,rho_all_x_y_plain,'k.')
hold on
plot(rho_all_x_y_plain,rho_all_x2_y_ort_x2,'go')
plot(rho_all_x_y_plain,rho_all_x2_x2_ort_y,'g.')

legend({'plain','(x2 , y ort x_2)','(x2 , x_2 ort y)'},'interpreter','none')
xlabel('correlation')
ylabel('correlation coeff')

h_s2(8)=subplot(2,4,8);
plot(rho_all_x_y_plain,rho_all_x_y_plain,'k.')
hold on
plot(rho_all_x_y_plain,rho_all_y_x2_ort_y,'ro')
plot(rho_all_x_y_plain,rho_all_y_y_ort_x2,'r.')
plot(rho_all_x_y_plain,rho_all_y_x_ort_x2,'mo')
plot(rho_all_x_y_plain,rho_all_y_x2_ort_x,'m.')
legend({'plain','(y , x2 ort y)','(y , y ort x2)','(y , x ort x2)','(y , x2 ort x)'},'interpreter','none')
xlabel('correlation')
ylabel('correlation coeff')

linkaxes(h_s2,'xy')
xlim([-1.1 1.1])
ylim([-1.1 1.1])
savefig(gcf,'correlations_all')
saveas(gcf,'correlations_all','tif')

%% still work in progress

%% combining pearson values to get hipp values... STILL NOT SURE HOW TO COMBINE
ort_hipp_x_y_atanh=(atanh(rho_all_x_y_ort_x)./2+atanh(rho_all_y_x_ort_y)./2); %x - y
ort_hipp_x_y_no_atanh=((rho_all_x_y_ort_x)./2+(rho_all_y_x_ort_y)./2);

ort_hipp_x_x2_no_atanh=-((rho_all_y_x2_ort_y)./2+(rho_all_x2_y_ort_x2)./2); % x_2 - y

figure
h_coh(1)=subplot(2,1,1);
plot(range_coherence,ort_hipp_x_y_atanh)
hold on
plot(range_coherence,ort_hipp_x_y_no_atanh)
hold on
plot(range_coherence,rho_all_x_y_plain,'k')
plot(range_coherence,ort_hipp_x_x2_no_atanh,'g')
title(['rho (ort hipp) vs rho plain ' num2str(B)])
legend({'hipp atanh y-x','hipp no atanh y-x','plain','hipp no atanh -(y-x2)'})
xlabel('coherence')
ylabel('correlation')
ylim([-1 1])

h_coh(2)=subplot(2,1,2);
plot(rho_all_x_y_plain,ort_hipp_x_y_no_atanh,'k')
title('rho all plain - rho ort hipp')
hold on
plot(rho_all_x_y_plain,ort_hipp_x_x2_no_atanh,'g')
plot(rho_all_x_y_plain,rho_all_y_x_ort_x2,'m')
xlabel('correlation')
ylabel('corr orth')
legend({'hipp(y - x ort y)','-(y -x2 ort y)','y x x2'})
ylim([-1 1])
linkaxes(h_coh,'x')

% estimating the slope.. bad estimation until we'll get nice plots
m=mean(diff(ort_hipp_x_y_no_atanh(1:end/2))./diff(rho_all_x_y_plain(1:end/2)))
m_2=mean(diff(ort_hipp_x_x2_no_atanh(end/2:end))./diff(rho_all_x_y_plain(end/2:end))')

    
    
    
    
    
    
    
    
    