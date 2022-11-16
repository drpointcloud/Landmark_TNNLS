function [V,divs,alphas,D1] = mmd(K,x_idx,tests)

K = (K + K.')/2; %ensure symmetric  (doesn't ensure PSD)

m = sum(x_idx); % sample size for X
n = size(K,1) - m; % sample size for Y

alphas = 1/m*x_idx - 1/n*(~x_idx);
div_mmd = alphas'*K*alphas;
divs = div_mmd;% div_mmd_max_sliced];

V = K*alphas; % witness function evaluations


if  nargin> 2 && tests > 1
    p1=zeros(tests,1);
    for t=1:tests
        new_idx = x_idx(randperm(n+m));
        [~,p1(t),~,~] = mmd(K,new_idx);
    end
    if nargout <= 4
        D1 = p1;
    end
else
    D1 = 0;
end


end
