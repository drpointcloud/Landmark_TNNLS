function B = sortOT(x,y)
% sortOT - Optimal transport in one dimension
% if the length of x = m == = length of y = n,
%   then n*B is a permutation matrix,
% otherwise it is a matrix whose row sums are 1/m
% and whose columns sums are 1/n

m = length(x);
n = length(y);
[~, idx1]=sort(x);
[~, idx2]=sort(y);    

if m==n
    B = sparse(idx1,idx2,1/m,m,m);
else 
    G = min((0:1/m:1)',(0:1/n:1));
    P = diff(diff(G,1,1),1,2);    
    B = zeros(m,n);
    B(idx1,idx2) = P;
end

end

