function B = sortOT(x,y,mu,nu)
% sortOT - Optimal transport in one dimension
% if the length of x = m == = length of y = n,
%   then n*B is a permutation matrix,
% otherwise it is a matrix whose row sums are 1/m
% and whose columns sums are 1/n

m = length(x);
n = length(y);
[~, idx1]=sort(x);
[~, idx2]=sort(y);    
 if  nargin<3
     mu = 1/m*ones(m,1);
     nu = 1/n*ones(n,1);    
 end

% 
% if m==n && nargin<3
%     B = sparse(idx1,idx2,1/m,m,m);
%     mu = 1/m*ones(m,1);
%     nu = 1/n*ones(n,1);    
% elseif nargin<3
%     G = min((0:1/m:1)',(0:1/n:1));
%     P = diff(diff(G,1,1),1,2);        
%     B = zeros(m,n);
%     B(idx1,idx2) = P;
%     mu = 1/m*ones(m,1);
%     nu = 1/n*ones(n,1);    
% 
% else
%     Px = cumsum([0;reshape(mu(idx1),[],1)]);
%     Py = cumsum([0;reshape(nu(idx2),[],1)]);
%     G = min(Px(:),Py(:)');
%     P = diff(diff(G,1,1),1,2);    
%     B = zeros(m,n);
%     B(idx1,idx2) = P;    
% end
% Based on  https://github.com/PythonOT/POT/blob/master/ot/lp/emd_wrap.pyx
    
    mu_sort = mu(idx1);
    nu_sort = nu(idx2);
    w_i = mu_sort(1);
    w_j = nu_sort(1);
    
    V = nan(m+n-1,1);
    IJ = nan(m+n-1,2);
    
    k = 1;
    i = 1;
    j = 1;
    while true
    if w_i < w_j || j == n
        V(k) = w_i;
        IJ(k,:) = [i,j];
        i = i + 1;
        if i > m
            break;
        end
        w_j = w_j - w_i;
        w_i = mu_sort(i);
    else
        V(k) = w_j;
        IJ(k,:) = [i,j];
        j = j + 1;
        if j > n
            break;
        end
        w_i = w_i - w_j;
        w_j = nu_sort(j);
    end
    k = k + 1;
    end
%       P1 = sparse(IJ(1:k,1),IJ(1:k,2),V(1:k),m,n); 
       B = sparse(idx1(IJ(1:k,1)),idx2(IJ(1:k,2)),V(1:k),m,n); 

end

