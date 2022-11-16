function [V,divs,alphas,D1] = L_MSKW(K,x_idx,tests)
% max_sliced_kernel_wasserstein_minfunc
% Inputs: K is kernel matrix, x_idx is logical index for one samples
% Purpose: compute the max-sliced kernel Wasserstein-2 metric
% \omega(*) = \sum_i \alpha_i K(z_i,*) 
% Outputs: div  (objective value), alpha (parameters of the solution)
% Austin J. Brockmeier (ajbrockmeier@gmail.com)
% 8/11/2020

%N = size(K,1);
% nMonte = 20; % number of random initializations
% if nargin==3
%     if isfield(params,'nMonte') 
%         nMonte = params.nMonte;
%     end
% end

% D_w = ||\omega||_2^2 
% 
% if iscell(method)
%     div_measure = method{1};
%     opt_method = method{2};
% else
%     div_measure = method;
%     opt_method = 'fastest'; 
% end


m = sum(x_idx); % sample size for X
n = size(K,1) - m; % sample size for Y

% K_X_Z  = A;
% K_Y_Z  = B;

K = (K + K.')/2; %ensure symmetric  (doesn't ensure PSD)
    K_X_Z  = K(  x_idx,  :);
    K_Y_Z  = K( ~x_idx, :);



if m==n  % assumes the X and Y are equal size
    landmark_divs = mean( (sort(K_X_Z) - sort(K_Y_Z)).^2 , 1);    
    %G = min((0:1/m:1)',(0:1/n:1));
    %P = diff(diff(G,1,1),1,2);
    %landmark_divs = mean(K_X_Z.^2,1) + mean(K_Y_Z.^2,1) - 2*sum((P'*sort(K_X_Z)).*sort(K_Y_Z) ,1);
else %if m is not equal to n, mass spliting applied for exact corresponding
    G = min((0:1/m:1)',(0:1/n:1));
    P = diff(diff(G,1,1),1,2);
    %landmark_divs = mean( (n*P'*sort(K_X_Z) - sort(K_Y_Z)).^2 , 1);
    %landmark_divs = sqrt( mean( (n*P'*sort(K_X_Z) - sort(K_Y_Z)).^2 , 1));
    landmark_divs = mean(K_X_Z.^2,1) + mean(K_Y_Z.^2,1) - 2*sum((P'*sort(K_X_Z)).*sort(K_Y_Z) ,1);
end


[max_val,idx] = max(landmark_divs,[],'omitnan');
div_max = sqrt(max(0,max_val));
%div_mean = mean(landmark_divs,'omitnan');
divs = div_max;%[div_max div_mean];

alphas = zeros(size(K,1),1);
alphas(idx) = 1;
V = K*alphas;

%%%%%%%%%%%%%%%%%%%%%%%%%%% slow %%%%%%%%%%%%%%%%%%%%%%%
% if  nargin >2 && tests >1
%     p1=zeros(tests,1);
%     for t=1:tests
%         new_idx = x_idx(randperm(n+m));
%         [~,p1(t,:),~,~] = L_MSKW(K,new_idx);
%     end
%     if nargout <= 4
%         %D1 = mean(p1<divs); % proportion of the shuffle less than D_22
%         %D2 = 1 - mean(D2>p2);
%         D1 = p1;
%     end
% else
%     D1 =0;
% end​
%%%%%%%%%%%%%%%%%%%%%%%%%%% slow %%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%% faster %%%%%%%%%%%%%%%%%%%%%%%
if nargin >2 && tests>1

    p2 = zeros(tests,1);
    K_tilde = sort(K);
    if m==n
        for t=1:tests
            rand_perm = x_idx(randperm(numel(x_idx)), :);
            p2(t,:) = sqrt(max(0,max(mean((K_tilde(rand_perm,:) - K_tilde(~rand_perm,:)).^2,1))));
        end
    else
        for t=1:tests
            rand_perm = x_idx(randperm(numel(x_idx)), :);
            KXZ = K_tilde(rand_perm,:);
            KYZ = K_tilde(~rand_perm,:);
            p2(t,:) = sqrt(max(0,max(mean(KXZ.^2,1) + mean(KYZ.^2,1) - 2*sum((P'*KXZ).*KYZ,1))));
        end
    end
    if nargout<=4
        D1 = p2;
    end
else
    D1 = 0;  
end
%%%%%%%%%%%%%%%%%%%%%%%%%%% faster %%%%%%%%%%%%%%%%%%%%%%%

end


%%



