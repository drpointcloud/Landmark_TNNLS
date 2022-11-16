
clc
clear
%%
load('/Users/yuksel/Documents/Dataset/EMNIST/emnist-mnist.mat');
rng(0);  % set the random seed
%%
% Requires minFunc_2012
%  addpath(genpath('path_to/minFunc_2012'))
%%
%% obtain a global median kernel bandwidh from a set of samples
y_sel = randperm(numel(dataset.test.labels),250);
x_sel = randperm(numel(dataset.train.labels),250);
X = dataset.train.images(x_sel,:);
Y = dataset.test.images(y_sel,:);
Z = double(cat(1,X,Y)); % combine two samples
[K,sigma] = gaussian_kernel(Z,'median');
%%
sample_size = 500;%round(100*linspace(1,5,5)); % Number of samples from p_X
nperms = 25;
p_missing = 0.75;%linspace(0,1,8);% prevalence is (1-p)/10
nmonte = 50;
classes = ['0'];%,'1','2','3','4','5','6','7','8','9'];
kernel_size = sigma*logspace(-1,1,10);
%%

methods = {
    %{@(K,x_idx,method) L_MSKB_one_side(K,x_idx,nperms), 'L-Bures-max',@(x) x};
    {@(K,x_idx) L_MSKW(K,x_idx,nperms), 'L-W2-max',@(x) x};
    {@(K,x_idx) mmd(K,x_idx,nperms), 'MMD',@(x) x};
    };
Nmthds = numel(methods);
Npmiss = numel(p_missing);
Nkernelsize = numel(kernel_size);
methods_name = arrayfun(@(i) methods{i}{2},(1:Nmthds),'uni',0);


%%
Div_value = zeros(numel(sample_size),numel(p_missing),numel(kernel_size),numel(classes),nmonte,Nmthds);
Betahat = zeros(numel(sample_size),numel(p_missing),numel(kernel_size),numel(classes),nmonte,Nmthds);
Witness = cell(size(Betahat));
P_at_10 = nan(numel(sample_size),numel(p_missing),numel(kernel_size),numel(classes),nmonte,Nmthds);

%% disp progress
aaa = 0;
perc = numel(sample_size)*numel(p_missing)*numel(kernel_size)*numel(classes);
timestart = tic;
progress = '..........';
%%
for sample_ii = 1:numel(sample_size)
    Nx = sample_size(sample_ii);
    Ny = round(1*Nx); % m=n or m\neq n

    for pmiss_ii = 1:numel(p_missing)
        p_with_missing = p_missing(pmiss_ii);

        for sigma_ii = 1:Nkernelsize
            sigma_size = kernel_size(sigma_ii);

            for l = classes%['0','1','2','3','4','5','6','7','8','9']
                label_ii = dataset.mapping(dataset.mapping(:,2)==l,1)+1;


                div_value = zeros(nmonte,Nmthds);
                betahat = zeros(nmonte,Nmthds);
                witness = cell(nmonte,Nmthds);
                p_at_10 = nan(nmonte,Nmthds);

                aaa=aaa+1; %to track progress
                parfor monte_ii = 1:nmonte

                    %X has very few of missing %Y has all characters
                    %density ratio p_Y(x)/p_X(x) should be very large for x in missing
                    y_sel = randperm(numel(dataset.test.labels),Ny);
                    not_label = dataset.mapping(dataset.mapping(:,2)==l,1);
                    subset = find(dataset.train.labels~=not_label);
                    x_sel1 = subset(randperm(numel(subset),round(Nx*p_with_missing)));
                    x_sel2 = randperm(numel(dataset.train.labels), Nx - round(Nx*p_with_missing));
                    x_sel = cat(1,x_sel1(:),x_sel2(:));
                    %[a,~]=hist(L_X,unique(dataset.train.labels(x_sel)))

                    X = dataset.train.images(x_sel,:);
                    L_X = dataset.train.labels(x_sel);
                    Y = dataset.test.images(y_sel,:);
                    L_Y = dataset.test.labels(y_sel);

                    %%
                    Z = double(cat(1,X,Y)); % combine two samples
                    x_idx = cat(1,ones(size(X,1),1),zeros(size(Y,1),1))==1; %bool indicator
                    [K,~] = gaussian_kernel(Z,sigma_size);

                    %% methods
                    for method_ii = 1:Nmthds

                        method_way = methods{method_ii}{1};
                        [V,divs,~,D1] =  method_way(K,x_idx);
                        betahat(monte_ii, method_ii) = mean(D1>=divs);
                        div_value(monte_ii, method_ii) = divs;
                        witness{monte_ii,method_ii} = V;
                        %end

                        %%
                        %for method_ii = 1:Nmthds

                        values = witness{monte_ii,method_ii}(~x_idx); %## works
                        [~,ordering] = sort(values,'descend');

                        auc1a = calcAUC(values,L_Y==not_label);
                        auc1b = calcAUC(-values,L_Y==not_label);

                        if auc1a>auc1b
                            auc(monte_ii,method_ii) = auc1a;
                        else
                            auc(monte_ii,method_ii) = auc1b;
                            ordering = ordering(end:-1:1);
                        end
                        p_at_10(monte_ii,method_ii) = mean(L_Y(ordering(1:numel(classes)))==not_label); %## works

                    end  %methods
                end% nmonte
                %1-ssize,2-pmiss,3-kernelsize,4-class,5-nmonte,6-methods
                Betahat(sample_ii, pmiss_ii,sigma_ii,label_ii,:,:) = betahat;
                Div_value(sample_ii, pmiss_ii,sigma_ii,label_ii,:,:) = div_value;
                Witness(sample_ii, pmiss_ii,sigma_ii,label_ii,:,:) = witness;
                P_at_10(sample_ii, pmiss_ii,sigma_ii,label_ii,:,:) = p_at_10;

                clc
                progress = progress_print(aaa,perc,timestart,progress);
           end % classes
        end % kernel size
    end % p_missing
end % sample size



%%
datestr(now)
linestyle= {':',':',':','-','-','-','-',':','--','-'};
markers = {'*','^','x','s','d'};
colors = {'m','b','g','r','k','y','c'}; %each color for a class: so maximum 7 classes
fontsize = 14;
linewidth = 3;
markersize = 10;
legendfontsize = 16;


%% power as a function of sample size
pow_alpha = [0.05 0.10 0.15];
%Betahat(ssize,pmiss,kernelsize,class,nmonte,methods)
betahat_ssize = Betahat(:,1,1,1,:,:); % choose a p_missing value

%to adjust subplot positions
n_row = 1; n_col = numel(pow_alpha); % # of columns and rows
fig_width = 0.8;fig_height= 0.65; %width and heigth of plots
%do not change this part
pad_w = (1-fig_width)/(n_col+1); pad_h = (1-fig_height)/(n_row+1);
axis_w = 1/n_col*fig_width; axis_h = 1/n_row*fig_height;

h = figure(2); clf; set(h,'WindowStyle','docked');
for i = 1:numel(pow_alpha)
    alpha_ii = pow_alpha(i);
    powhat_ssize = mean(betahat_ssize<alpha_ii,5);

    %to adjust subplot positions:do not change
    row = floor((i-1)/n_col)+1; col = mod(i-1,n_col)+1;
    axis_l = axis_w*(col-1) + pad_w*col; axis_b = axis_h*(n_row-row) + pad_h*(n_row-row+1);

    kk =0;
    subplot('position', [axis_l axis_b axis_w axis_h] )
    for method_ii=1:numel(methods)
        kk=kk+1;
        plot(powhat_ssize(:,:,:,:,:,method_ii),':','linewidth',linewidth,...
            'Marker',markers{kk},'MarkerSize',markersize);hold on
    end

    t = title(['Rejection rate at \alpha  =',sprintf('%0.2f\n',alpha_ii)]);
    set(t,'position',get(t,'position')-[0 axis_b/10 0])
    xlabel('Sample size (N)');
    xticks(1:2:numel(sample_size))
    set(gca,'XTickLabel',sprintf('%d\n',sample_size(1:2:end)),'fontsize',fontsize)
    legend(methods_name,'FontSize',legendfontsize,'Location','southeast','Interpreter','latex');legend('boxoff')
    grid on
end

%% power as a function of p-missing
pow_alpha = [0.05 0.10 0.15];
%Betahat(ssize,pmiss,kernelsize,class,nmonte,methods)
betahat_pmiss = Betahat(1,:,1,1,:,:);

%to adjust subplot positions
n_row = 1; n_col = numel(pow_alpha); % # of columns and rows
fig_width = 0.8;fig_height= 0.65; %width and heigth of plots
%do not change this part
pad_w = (1-fig_width)/(n_col+1); pad_h = (1-fig_height)/(n_row+1);
axis_w = 1/n_col*fig_width; axis_h = 1/n_row*fig_height;


h = figure(1); clf; set(h,'WindowStyle','docked');
for i = 1:numel(pow_alpha)
    alpha_ii = pow_alpha(i);
    powhat_pmiss = mean(betahat_pmiss<alpha_ii,5); %mean across nmonte

    %to adjust subplot positions:do not change
    row = floor((i-1)/n_col)+1; col = mod(i-1,n_col)+1;
    axis_l = axis_w*(col-1) + pad_w*col; axis_b = axis_h*(n_row-row) + pad_h*(n_row-row+1);

    kk =0;
    subplot('position', [axis_l axis_b axis_w axis_h] )
    for method_ii=1:numel(methods)
        kk=kk+1;
        plot(powhat_pmiss(:,:,:,:,:,method_ii),':','linewidth',linewidth,...
            'Marker',markers{kk},'MarkerSize',markersize); hold on
    end

    t = title(['Rejection rate at \alpha  =',sprintf('%0.2f\n',alpha_ii)]);
    %set(t,'position',get(t,'position')-[0 axis_b/10 0])
    xlabel('Prevalence of underrepresented digit');
    xticks(1:2:numel(p_missing))
    set(gca,'XTickLabel',sprintf('%0.2f\n',(1-p_missing(1:2:end))/10),'fontsize',fontsize)
    legend(methods_name,'FontSize',legendfontsize,'Location','southeast','interpreter','latex');legend('boxoff')
    grid on
end


%% power as a function of kernel size
%matlab default plots colors
colors = [0.00, 0.45, 0.74;  %blue      	
          0.85, 0.32, 0.10;  %red 
          0.93, 0.69, 0.13]; %yellow

%Betahat(ssize,pmiss,kernelsize,class,nmonte,methods)
betahat_ksize = Betahat(1,1,:,1,:,:); % choose a sample size and a pmiss

%to adjust subplot positions: do not change
n_row = 1; n_col = numel(pow_alpha); % # of columns and rows
fig_width = 0.8;fig_height= 0.65; %width and heigth of plots
%do not change this part
pad_w = (1-fig_width)/(n_col+1); pad_h = (1-fig_height)/(n_row+1);
axis_w = 1/(n_col)*fig_width; axis_h = 1/n_row*fig_height;

h = figure(3); clf; set(h,'WindowStyle','docked');
for i = 1:numel(pow_alpha)
    alpha_ii = pow_alpha(i);
    powhat_ssize = mean(betahat_ksize<alpha_ii,5);

    %to adjust subplot positions: do not change
    row = floor((i-1)/n_col)+1; col = mod(i-1,n_col)+1;
    axis_l = axis_w*(col-1) + pad_w*col; axis_b = axis_h*(n_row-row) + pad_h*(n_row-row+1);

    kk =0;
    subplot('position', [axis_l axis_b axis_w axis_h] )
    for method_ii=1:numel(methods)
        kk=kk+1;
        plot(squeeze(powhat_ssize(:,:,:,:,:,method_ii)),':','linewidth',linewidth,'color',colors(kk+1,:),...
            'Marker',markers{kk+1},'MarkerSize',markersize);hold on
    end

    t = title(['Rejection rate at \alpha  =',sprintf('%0.2f\n',alpha_ii)]);
    set(t,'position',get(t,'position')-[0 axis_b/2 0])
    xlabel('Kernel size (\sigma)');
    xticks(1:2:numel(kernel_size))
    set(gca,'XTickLabel',sprintf('%0.0f\n',kernel_size(1:2:end)),'fontsize',fontsize)
    legend(methods_name,'FontSize',legendfontsize,'Location','southeast','Interpreter','latex');legend('boxoff')
    %legend('')
    grid on
end

%sgtitle('Sample size (N=500), prevalence:0.025, digit:"0"')

%% plot p@10 as a function of pmiss,sample size, and kernel size
% 1ssize,2pmiss,3kernelsize,4class,5nmonte,6methods
mean_p10 = mean(P_at_10(:,:,:,1,:,:),5);

%to adjust subplot positions:
n_row = 1; n_col = 3; % # of columns and rows
fig_width = 0.8;fig_height= 0.65; %width and heigth of plots
%do not change this part
pad_w = (1-fig_width)/(n_col+1); pad_h = (1-fig_height)/(n_row+1);
axis_w = 1/n_col*fig_width; axis_h = 1/n_row*fig_height;

%x_axis:p_miss
i=1; %to adjust subplot positions: do not change
row = floor((i-1)/n_col)+1;col = mod(i-1,n_col)+1;
axis_l = axis_w*(col-1) + pad_w*col;axis_b = axis_h*(n_row-row) + pad_h*(n_row-row+1);

h = figure(4); clf; set(h,'WindowStyle','docked');
kk=0;
subplot('position', [axis_l axis_b axis_w axis_h] )
for method_ii=1:numel(methods)
    kk=kk+1;
    plot(mean_p10(end,:,1,1,1,method_ii),'linestyle',linestyle{kk},...
        'linewidth',3,'Marker',markers{kk},'MarkerSize',10);hold on
end

%ylim([0 1])
t = title(['Precision@10 (N=',sprintf('%d)\n',sample_size(end))]);
set(t,'position',get(t,'position')-[0 axis_b/50 0])
xlabel('Prevalence of underrepresented class'); %ylabel('Precision at 10');
xticks(1:numel(p_missing))
set(gca,'XTickLabel',sprintf('%0.3f\n',(1-p_missing(1:end))/10),'fontsize',fontsize)
legend(methods_name,'FontSize',legendfontsize,'Location','southeast','interpreter','latex');legend('boxoff')
grid on

%x_axis:sample size
i=2; %to adjust subplot positions: do not change
row = floor((i-1)/n_col)+1;col = mod(i-1,n_col)+1;
axis_l = axis_w*(col-1) + pad_w*col;axis_b = axis_h*(n_row-row) + pad_h*(n_row-row+1);

kk=0;
p_miss = 1;
subplot('position', [axis_l axis_b axis_w axis_h] )
for method_ii=1:numel(methods)
    kk=kk+1;
    plot(mean_p10(:,p_miss,1,1,1,method_ii),'linestyle',linestyle{kk},...
        'linewidth',linewidth,'Marker',markers{kk},'MarkerSize',10);hold on
end

t = title(['Precision@10 (Prevalence = ',sprintf('%0.3f)\n',(1-p_missing(p_miss))/10)]);
set(t,'position',get(t,'position')-[0 axis_b/50 0])
xlabel('Sample size (N)'); %ylabel('Precision at 10');
xticks(1:numel(sample_size))
set(gca,'XTickLabel',sprintf('%d\n',sample_size),'fontsize',fontsize)
legend(methods_name,'FontSize',legendfontsize,'Location','southeast','interpreter','latex');legend('boxoff')
grid on

%x_axis:kernel size
i=3; %to adjust subplot positions: do not change
row = floor((i-1)/n_col)+1;col = mod(i-1,n_col)+1;
axis_l = axis_w*(col-1) + pad_w*col;axis_b = axis_h*(n_row-row) + pad_h*(n_row-row+1);

kk=0;
p_miss = 1;
subplot('position', [axis_l axis_b axis_w axis_h] )
for method_ii=1:numel(methods)
    kk=kk+1;
    plot(squeeze(mean_p10(end,p_miss,:,1,1,method_ii)),'linestyle',linestyle{kk},...
        'linewidth',linewidth,'Marker',markers{kk},'MarkerSize',10);hold on
end

t = title(['Precision@10 (Prevalence = ',sprintf('%0.3f)\n',(1-p_missing(p_miss))/10)]);
set(t,'position',get(t,'position')-[0 axis_b/50 0])
xlabel('Kernel size (\sigma)');
xticks(1:2:numel(kernel_size))
set(gca,'XTickLabel',sprintf('%0.1f\n',kernel_size(1:2:end)),'fontsize',fontsize)
legend(methods_name,'FontSize',legendfontsize,'Location','southeast','interpreter','latex');legend('boxoff')
grid on

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Fashion_Mnist  CIFAR-10
% '0'	'T-shirt/top'   airplane
% '1'	'Trouser'       automobile
% '2'	'Pullover'      bird
% '3'	'Dress'         cat
% '4'	'Coat'          deer
% '5'	'Sandal'        dog
% '6'	'Shirt'         frog
% '7'	'Sneaker'       horse
% '8'	'Bag'           ship
% '9'	'Ankle boot'    truck
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Print the progress
function string = progress_print(aaa,perc,timestart,string)
tEnd = toc(timestart);
percentage = aaa/perc*100;
remt = 100*tEnd/percentage -tEnd;
if fix(percentage/10) ~= 0
    string(fix(percentage/10)) = '*';
end
fprintf('\n[%s]', string);
fprintf(' %0.1f%% is completed...\n',percentage)
disp(['elapsed time:',datestr(datenum(0,0,0,0,0,tEnd),'DD:HH:MM:SS\n'),...
    'est rem time:',datestr(datenum(0,0,0,0,0,remt),'DD:HH:MM:SS')])
%fprintf('nM(%d/%d) ',mod(ceil(aaa/Nmethods),nMonte),nMonte)
%fprintf('Ep(%d/%d)\n ',ceil(aaa/nMonte/Nmethods),perc/Nmethods/nMonte)
end

















%%
