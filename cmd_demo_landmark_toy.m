clear
clc

%%
rng(0);
n = 1000;
nperms = 1;


%%
methods_landmark = {
    {@(K,x_idx) L_MSKW(K,x_idx,nperms),'L-W2-max','L-W2',@(x) x};
    %{@(K,x_idx,method) L_MSKB_one_side(K,x_idx,nperms),'L-Bures','L-Bures',@(x) x};
    };

methods1 = {
    {@(K,x_idx) mmd(K,x_idx), 'MMD','MMD',@(x) x};
    };

methods = cat(1,methods_landmark,methods1);
methods_name = arrayfun(@(i) methods{i}{3},(1:numel(methods)),'uni',0);

%% Create data
%Create uniform with circular support using rejection sampling
Z = 2*(rand(4*5*n,2)-0.5);
idx = sum(Z.^2,2)<=1;
Z = Z(idx,:);
% Circle offsets
offsets = 1.5*[1 0 ; 0 0; cos(pi/3) sin(pi/3)];
X = Z(1:n,:) + offsets(randi(3,n,1), :); % X ~ \mu is uniformly across the support
%Y ~ \nu has mass [0, 1/3, 2/3] in each circle

m = round(n*(1));%  m=n or m\neq n
Y = Z(m+1:2*m,:) + offsets(2+(rand(m,1)>1/3), :);


%%
Witness = cell(size(numel(methods)));
Alphas = Witness;
%%

% Pool the samples and make the kernel matrix
x_idx = cat(1,ones(size(X,1),1),zeros(size(Y,1),1))==1; %indicator for X
Z = cat(1,X,Y);
[K,sigma]= gaussian_kernel(Z,'median');


% Define the kernel for test points
kappa = @(x,y,sigma) exp(-( sum(x.^2,2) + sum(y.^2,2).' -2*x*y.')/2/sigma^2);
% Make a grid of test points for contours
[X_test,Y_test]=meshgrid(linspace(min(Z(:,1)),max(Z(:,1)),40),linspace(min(Z(:,2)),max(Z(:,2)),40));


%% Landmark Method(s)
offset = 0;
for method_ii = 1:numel(methods_landmark)
    method_way = methods_landmark{method_ii}{1};
    [V,divs,alphas,D1] =  method_way(K,x_idx);
    Witness{method_ii+offset} = V;
    Alphas{method_ii+offset} = alphas;
    W = Witness{method_ii+offset};

    omega_X_ge_Y = W;%W(W<0);%W(:,1);
    omega_X_le_Y = W;%W(W>0);%W(:,2);
    % Get coefficients for defining witness functions
    alpha1 = alphas;
    alpha2 = alphas;

    % Sort the witness function evaluations by their magnitude
    % Points from X ~ \mu
    [~,pi_ge] = sort(omega_X_ge_Y(x_idx).^2,'descend');
    [~,pi_le] = sort(omega_X_le_Y(x_idx).^2,'descend');
    % Points from Y ~ \nu
    [~,sigma_ge] = sort(omega_X_ge_Y(~x_idx).^2,'descend');
    [~,sigma_le] = sort(omega_X_le_Y(~x_idx).^2,'descend');

    % Get magnitude of witness function evaluations on test points
    omega_X_ge_Y_test = 0*X_test;
    omega_X_ge_Y_test(:) = (kappa([X_test(:),Y_test(:)],Z,sigma)*alpha1).^2;
    omega_X_le_Y_test = 0*X_test;
    omega_X_le_Y_test(:) = (kappa([X_test(:),Y_test(:)],Z,sigma)*alpha2).^2;

    k = 10; % Number of witness points top-k
    s = 2; % two figures (bures is two sided, so 4 figures)

    h = figure(10+method_ii);clf;set(h,'WindowStyle','docked') %dock the figure
    plotfigures(X,Y,s,k,X_test,Y_test,omega_X_ge_Y_test,omega_X_le_Y_test,sigma_ge,sigma_le,pi_ge,pi_le,methods_name{method_ii+offset})
end

%% Baseline Method(s)
offset = offset+numel(methods_landmark);
for method_ii = 1:numel(methods1)
    method_way = methods1{method_ii}{1};
    method = methods1{method_ii}{2};
    [V,~,alphas] =  method_way(K,x_idx);
    Witness{method_ii+offset} = V;
    Alphas{method_ii+offset} = alphas;
    W = Witness{method_ii+offset};



    W1 = W; W1(W1<0)=0; %positive
    W2 = W; W2(W2>0)=0; %negative

    omega_X_ge_Y = W1;%W(W<0);%W(:,1);
    omega_X_le_Y = W2;%W(W>0);%W(:,2);
    % Get coefficients for defining witness functions
    alpha1 = alphas;
    alpha2 = alphas;

    % Sort the witness function evaluations by their magnitude
    % Points from X ~ \mu
    [~,pi_ge] = sort(omega_X_ge_Y(x_idx).^2,'descend');
    [~,pi_le] = sort(omega_X_le_Y(x_idx).^2,'descend');
    % Points from Y ~ \nu
    [~,sigma_ge] = sort(omega_X_ge_Y(~x_idx).^2,'descend');
    [~,sigma_le] = sort(omega_X_le_Y(~x_idx).^2,'descend');

    % Get magnitude of witness function evaluations on test points
    omega_X_ge_Y_test = 0*X_test;
    omega_X_ge_Y_test(:) = (kappa([X_test(:),Y_test(:)],Z,sigma)*alpha1);
    omega_X_le_Y_test = 0*X_test;
    omega_X_le_Y_test(:) = (kappa([X_test(:),Y_test(:)],Z,sigma)*alpha2);

    k = 10; % Number of witness points top-k
    s = 4; %two figures (bures is two sided, so 4 figures)

    h = figure(20+method_ii);clf;set(h,'WindowStyle','docked') %dock the figure
    plotfigures(X,Y,s,k,X_test,Y_test,omega_X_ge_Y_test,omega_X_le_Y_test,sigma_ge,sigma_le,pi_ge,pi_le,methods_name{method_ii+offset})
end



%%  PLOT FIGURES
function plotfigures(X,Y,s,k,X_test,Y_test,omega_X_ge_Y_test,omega_X_le_Y_test,sigma_ge,sigma_le,pi_ge,pi_le,current_method)
% Color-blind safe two color scheme
%colors=[31,120,180;178,223,138]/255;

marker_size_x = 6;
marker_size_y = 6;
marker_size_c = max(marker_size_x,marker_size_y) + 4;
if size(X,1)==size(Y,1)
    %dim = get(h,'Position') -  [0.1 h.Position(2) 0 0];
    str = [sprintf('%s   ',current_method), '  $(m=n)$'];
    %annotation('textbox',dim,'String',str,'FitBoxToText','on','FontSize',15);
else
    str = [sprintf('%s   ',current_method), '  $(m\neq n)$'];
end

for ii = 1:s
    subplot(s/2,2,ii)
    plot(X(:,1),X(:,2),'oc','markersize',marker_size_x);
    hold all;
    plot(Y(:,1),Y(:,2),'+r','markersize',marker_size_y);
    set(gca,'fontsize',18);axis tight; axis equal; axis off
    %set(gca,'visible','off')

    switch ii
        case 1
            contour(X_test,Y_test,omega_X_ge_Y_test)
            plot(X(pi_ge(1:k),1),X(pi_ge(1:k),2),'ok','markersize',marker_size_c)
            witness_points = '$x_{\grave{\pi}(1)},\ldots,x_{\grave\pi(10)}$';
            h=legend('$X\sim \mu $','$Y\sim\nu$','$\omega^2_{\mu>\nu}$',...
                'FontSize',13,'interpreter','latex','location','southwest');
            title(h,str,'interpreter','latex','FontWeight','normal')
            %legend('boxoff')
            set(h,'position',[ 0.4545    0.7996    0.1509    0.1340])
        case 2
            contour(X_test,Y_test,omega_X_ge_Y_test)
            plot(Y(sigma_ge(1:k),1),Y(sigma_ge(1:k),2),'ok','markersize',marker_size_c)
            witness_points = '$y_{\grave{\sigma}(1)},\ldots,y_{\grave\sigma(10)}$';
        case 3
            contour(X_test,Y_test,omega_X_le_Y_test)
            plot(Y(sigma_le(1:k),1),Y(sigma_le(1:k),2),'ok','markersize',marker_size_c)
            witness_points = '$y_{\acute{\sigma}(1)},\ldots,x_{\acute\sigma(10)}$';
            h=legend('$X\sim \mu $','$Y\sim\nu$','$\omega^2_{\mu<\nu}$','interpreter','latex','location','southwest');
            set(h,'position',[  0.4409    0.3142    0.1509    0.1340])
            legend('boxoff')
        case 4
            contour(X_test,Y_test,omega_X_le_Y_test)
            plot(X(pi_le(1:k),1),X(pi_le(1:k),2),'ok','markersize',marker_size_c)
            witness_points = '$x_{\acute{\pi}(1)},\ldots,x_{\acute\pi(10)}$';
    end
    %plotcoordinates = get(gca,'position');
    text(-4.1,-6.2,witness_points,'fontsize',20,'interpreter','latex')
end
colormap(flipud(gray))
end



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
