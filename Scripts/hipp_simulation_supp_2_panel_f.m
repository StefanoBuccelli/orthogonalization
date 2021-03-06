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
savefig(gcf,'supp_fig_2_panel_f')
saveas(gcf,'supp_fig_2_panel_f','tif')

