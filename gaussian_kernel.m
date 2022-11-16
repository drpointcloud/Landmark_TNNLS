function [K,sigma] = gaussian_kernel(X,kernel_size)
%gaussian_kernel Creates the kernel matrix using
%                a radial basis function (Gaussian kernel)
% X is n-by-d matrix with n data points and d dimensions
% kernel_size is either a string {'median','info'}
%               or a scalar or vector indicating length-scale of each
%               dimension

[n,d] = size(X);
% Assume Gaussian kernel
%sigma2s = logspace(-4,4,30);
rbf2 = @(d2,sigma) exp(-d2/(2*sigma^2));
% Rely on squared Euclidean distances
D2 = max(0,  -2*(X*X.') + sum(X.^2,2) + sum(X.^2,2).');

switch kernel_size
    case 'median'
        sigma = median(reshape(sqrt(D2) + sparse(1:n,1:n,nan,n,n),[],1), 'omitnan');
    case 'info'
        search_limits = quantile(...
            reshape(sqrt(D2) + sparse(1:n,1:n,nan,n,n),[],1),...
            [0.05 0.95]); % search limits as 5th and 95th quantile
        % of distances
        sigmas = logspace(log10(search_limits(1)),...
            log10(search_limits(2)),20);
        infoval = nan*sigmas;
        for ii=1:numel(sigmas)
            sigma= sigmas(ii);
            K = rbf2(D2,sigma);
            infoval(ii) = matrix_info_cosine(K); % requires informativeness code
        end
        [~,best_idx] = max(infoval);
        sigma = sigmas(best_idx);
    otherwise
        assert(isfloat(kernel_size) && ...
            (numel(kernel_size)== 1 || numel(kernel_size)==size(X,2)))
        sigma = kernel_size; % assume it is a floating point number
end
if numel(sigma)==1
    K = rbf2(D2,sigma);
else
    X = X*spdiags(sigma(:),0,d,d);
    % Rely on squared Euclidean distances
    D2 = max(0,  -2*(X*X.') + sum(X.^2,2) + sum(X.^2,2).');
    K = rbf2(D2,1);
end
K  = 0.5*(K+K');
end

