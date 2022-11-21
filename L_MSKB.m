function [V,divs,alphas,D1] = L_MSKB_one_side(K,x_idx,tests)
% one_side_max_bures_minfunc
% Inputs: A = D_\mu^0.5 K_XZ  and C= D_\nu^0.5 K_YZ
%           or vice versa
%       K is full kernel matrix
% Purpose: compute one side of the max sliced kernel Bures
% \omega(*) = \sum_i \alpha_i K(z_i,*) 
% Outputs: div  (objective value), alpha (parameters of the solution)
% Austin J. Brockmeier (ajbrockmeier@gmail.com)
% 8/11/2020

% N = size(K,1);
% 
% nMonte = 2; % number of random initializations
% max_iter = N^2;
% stop_delta = 1e-6;
% verbose = false;
% if nargin==4
%     if isfield(params,'verbose') 
%         verbose = params.verbose;
%     end
%     if isfield(params,'stop_delta') 
%         stop_delta = params.stop_delta;
%     end
%     if isfield(params,'max_iter') 
%         max_iter = params.max_iter;
%     end
%     if isfield(params,'nMonte') 
%         nMonte = params.nMonte;
%     end
% end
m = sum(x_idx); % sample size for X
n = size(K,1) - m; % sample size for Y

K = (K + K.')/2; % ensure symmetric
K_X_Z  = K(  x_idx,  :); % Z = X union Y
K_Y_Z  = K( ~x_idx, :);

A = sqrt(1/n)*K_Y_Z; % D_\nu^\frac{1}{2}  K_{YZ}
B = sqrt(1/m)*K_X_Z; % D_\mu^\frac{1}{2}  K_{XZ}

%%
landmark_divs = abs(sqrt(sum(A.^2)) -  sqrt(sum(B.^2)));

[div_max , idx] = max(landmark_divs,[],'omitnan');
divs = div_max;%[div_max div_mean];

alphas = zeros(size(K,1),1);
alphas(idx) = 1;
V = K*alphas;

if  nargin> 2 && tests > 1
    p1=zeros(tests,1);
    for t=1:tests
        new_idx = x_idx(randperm(n+m));
        [~,p1(t,:),~,~] = L_MSKB_one_side(K,new_idx);
    end
    if nargout <= 4
        D1 = p1;
        
    end
else
    D1=0;
end





