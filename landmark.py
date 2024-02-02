# landmark
import numpy as np
from scipy import sparse
# from utils import *
import ot as ot
# from utils import gaussian_kernel


#######



def LMSKW(X, Y, tests=0):

    if np.sum((Y == 0) | (Y == 1)) == np.size(Y):
        x_idx = Y 
        K = X 
        m = np.sum(x_idx) # 1
        n = K.shape[0] - m
    else:
        m = X.shape[0] #5000
        n = Y.shape[0]
        d = X.shape[1] # 784*2
        
        x_idx = np.array(range(m+n)) < m 
        
        # d = X_tr.shape[1]
        _, sigmas, div, _ = gauss_rff_select_lmskb(X, Y, p = m+n-1)

        sigma = sigmas[np.argmax(div)] # scalar
        # [X, Y] = n_size, 784*2*2
        K, _ = gaussian_kernel(np.concatenate((X, Y), axis=0), sigma) #
    

    # K = (K + np.transpose(K)) / 2  # ensure symmetric  (doesn't ensure PSD)
    # K  = K*diag(diag(K).^(-1)); % normalize
    # K  = K*np.diagonal(np.diagonal(K)**-1) # normalize
    K_X_Z = K[x_idx, :]
    K_Y_Z = K[~x_idx, :]
    
    # print((K_X_Z - K_Y_Z).sum())
    # import matplotlib.pyplot as plt
    # plt.imshow(K, cmap='viridis')
    # plt.colorbar()   
        
    if m == n:  # assumes the X and Y are equal size
        # landmark_divs = mean( (sort(K_X_Z) - sort(K_Y_Z)).^2 , 1);
        landmark_divs = np.mean(np.square(np.sort(K_X_Z, axis=0) - np.sort(K_Y_Z, axis=0)), axis=0)
    else:  # if m is not equal to n, mass splitinßg applied for exact corresponding
        m_part = linear_space(0, 1, 1 / m)  # linspace(start,stop,step)
        n_part = linear_space(0, 1, 1 / n)
        # np.linspace(start, stop, int((stop - start) / step + 1))
        G = np.minimum(np.atleast_2d(m_part).T, np.atleast_2d(n_part))
        P = np.diff(np.diff(G, axis=0), axis=1)  # matlab: diff(data,nth order,dimention)
        # landmark_divs = mean(K_X_Z.^2,1) + mean(K_Y_Z.^2,1) - 2*sum((P'*sort(K_X_Z)).*sort(K_Y_Z) ,1);
        
        landmark_divs = np.mean(np.square(K_X_Z), axis=0) + np.mean(np.square(K_Y_Z), axis=0) - 2 * np.sum(
            (P.T @ np.sort(K_X_Z)) * np.sort(K_Y_Z))

    idx = np.unravel_index(np.argmax(landmark_divs, axis=None), landmark_divs.shape)
    max_val = np.nanmax(landmark_divs)
    div_max = np.sqrt(np.maximum(0, max_val))
    # div_mean = mean(landmark_divs,'omitnan')
    # divs = div_max;#%[div_max div_mean]
    alphas = np.zeros((K.shape[0], 1))
    alphas[idx] = 1
    V = K @ alphas
    # return div_max


    p_val = 0
    if (tests > 1):
        shuffled_tests = np.zeros((tests, 1))
        K_tilde = np.sort(K, axis=0)
        if m == n:
            for t in range(tests):
                rand_perm = x_idx[np.random.permutation(len(x_idx))]  # bool
                # p2(t,:) = sqrt(max(0,max(mean((K_tilde(rand_perm,:) - K_tilde(~rand_perm,:)).^2,1))));
                shuffled_tests[t] = np.sqrt(np.maximum(0, np.max(
                    np.mean(np.square(K_tilde[rand_perm, :] - K_tilde[~rand_perm, :]), axis=0))))
        else:
            for t in range(tests):
                rand_perm = x_idx[np.random.permutation(len(x_idx))]
                KXZ = K_tilde[rand_perm, :]
                KYZ = K_tilde[~rand_perm, :]
                # sqrt(max(0,max(mean(KXZ.^2,1) + mean(KYZ.^2,1) - 2*sum((P'*KXZ).*KYZ,1))));
                shuffled_tests[t] = np.sqrt(np.maximum([0], np.max(
                    np.mean(np.square(KXZ), axis=0) + np.mean(np.square(KYZ), axis=0) - 2 * np.sum(
                        (np.transpose(P) * KXZ) @ KYZ, axis=0))))

        p_val = np.mean(shuffled_tests >= div_max)
        # p_vals = np.array(p_val)
    return p_val, landmark_divs, V, alphas








def LMSKW_deflation(X, Y, k=10, topK=0, overlap_thresh=1/np.sqrt(2), indep = True):
    """ Landmark max-sliced kernel Wasserstein performed  
    repeatedly doing optimal transport after each iteration
    indep:  deflation either on test or training samples
    test: dependent 
    training: independent 
    """
    # X (n_data, n_image * 2)
    # Y (n_data, n_image * 2)

    # np.sum((X_te == 0) | (X_te == 1)) == np.size(X_te) # check whether all element of matrix is true
    if np.sum((Y == 0) | (Y == 1)) == np.size(Y):
        x_idx = Y 
        K = X 
        m = np.sum(x_idx) # 1
        n = K.shape[0] - m
    else:
        m = X.shape[0] # m
        n = Y.shape[0] # n
        d = X.shape[1] # 784*2
        
        x_idx = np.array(range(m+n)) < m 
        
        # d = X_tr.shape[1]
        _, sigmas, div, _ = gauss_rff_select_lmskb(X, Y, p = m+n-1)
        
        
        sigma = sigmas[np.argmax(div)] # scalar
        print('index=',np.argmax(div)+1,'|',len(sigmas),' sigma =',sigma)
        # [X, Y] = n_size, 784*2*2
        K, _ = gaussian_kernel(np.concatenate((X, Y), axis=0), sigma) #
        
        # import matplotlib.pyplot as plt
        # plt.imshow(K, cmap='viridis')
        # plt.colorbar()        
        
    if topK == 0:
        topK = m
        
    K0 = np.copy(K) 
    Ls = np.full([k, 1], np.nan)
    D = np.full([k,1], np.nan)
    pi_X = []
    # K_XX = K[x_idx, x_idx]
    # K_YY = K[~x_idx, ~x_idx]
    # K_XY = K[x_idx, ~x_idx]
    
    for i in range(0, k):

        _, Divs, V, alphas= LMSKW(K,x_idx)
        # idx = np.unravel_index(np.argmax(Divs, axis=None), Divs.shape)
        # div_max = np.maximum(0, np.nanmax(Divs))
        # i_star = np.nonzero(alphas)
        
        i_star = np.argmax(alphas)
        D[i] = Divs[:, np.newaxis][i_star]
        # if np.max(K0[i_star,Ls[1:i-1]]) > overlap_thresh*K0[i_star, i_star]:
        #     print("Too much overlap")
            
        K_Zz = K[:,i_star]
        K_zz = K0[i_star, i_star]
        K_Xz = K_Zz[x_idx]
        K_Yz = K_Zz[~x_idx]
        Ls[i] = i_star
        
        # get the trasport plan using POT
        P = ot.emd_1d(V[x_idx], V[~x_idx], dense=False)
        
        
        if indep:
            # deflation on independent samples
            K_Xz_u = P/np.sum(P, axis=1)@K_Yz
            K_Xz_hat = K_Xz - K_Xz_u
            K_XZ = K[x_idx,:]
            K_XZ = K_XZ - 1/K_zz * (K_Xz_hat[:,np.newaxis]*K0[np.newaxis,i_star,:])
            K[x_idx,:] = K_XZ
        else:
            # deflation on dependent samples
            K_Yz_u = P.T/np.sum(P, axis=1)@K_Xz
            K_Yz_hat = K_Yz - K_Yz_u
            K_YZ = K[~x_idx,:]
            K_YZ = K_YZ - 1/K_zz * (K_Yz_hat[:,np.newaxis]*K0[np.newaxis,i_star,:])
            K[~x_idx,:] = K_YZ
        
        
        div_after = np.sqrt(np.maximum(0,np.mean(K[x_idx,i_star]**2) +np.mean(K[~x_idx,i_star]**2) - 2*K[x_idx,i_star, np.newaxis].T @ P @ K[~x_idx,i_star, np.newaxis]))
        print("Divergence (before and after): %s %s"  % (np.round(D[i], decimals=6),div_after[0]))
        
        # sort witness function in descending order and get the indices 
        pi1 = np.argsort(V[x_idx,:], axis=0)[::-1] # mx1
        pi_X.append(pi1[0:topK])
        
    return Ls, D, pi_X







def linear_space(self, start, stop, step=1.):
    """
    Like np.linspace but uses step instead of num
    This indcludes the stop, so if start=1, stop=3, step=0.5
    Output is: array([1., 1.5, 2., 2.5, 3.])
    """
    return np.linspace(start, stop, int((stop - start) / step + 1))



def gaussian_kernel(X, kernel_size=None):  
    
    # X = (n_size * 2, d)
    # Assume Gaussian kernel
    def rbf2(D2, kernel_size):
       return np.exp(-D2 / (2 * kernel_size ** 2))
   
    n, _ = X.shape  # [n,d] = size(X);
    # kernel_size = np.logspace(-4,4,30) # %kernel_size = logspace(-4,4,30);
    # D2 = max(0,  -2*(X*X.') + sum(X.^2,2) + sum(X.^2,2).'); # % Rely on squared Euclidean distances
    
    D2 = np.maximum(0, -2 * (X @ X.T) 
                    + np.sum(X**2, axis=1, keepdims=True) 
                    + (np.sum(X**2, axis=1, keepdims=True)).T) # n_size * 2, n_size * 2
    
    # find the median kernel size
    if kernel_size is None:
        kernel_size = np.nanmedian(np.ravel(np.sqrt(D2) + sparse.spdiags(np.nan, 0, n, n)))
        #kernel_size = median(reshape(sqrt(D2) + sparse(1:n,1:n,nan,n,n),[],1), 'omitnan');

    

    if  isinstance(kernel_size, float): #kernel_size.size == 1:
        K = rbf2(D2, kernel_size) # n_size * 2, n_size * 2  
    elif isinstance(kernel_size, int):
        K = rbf2(D2, kernel_size) # n_size * 2, n_size * 2
    else:
        # X = X * sparse.spdiags(kernel_size, 0, d, d)
        # # Rely on squared Euclidean distances
        # D2 = np.maximum(np.zeros((n, n)), -2 * (X * np.transpose(X)) + np.sum(np.square(X), axis=1) + np.transpose(
        #     np.sum(np.square(X), axis=1)))
        # K = rbf2(D2, 1)
        # K = np.exp(-D2 / (2 * kernel_size ** 2))
        pass

    K = (K + np.transpose(K)) / 2 # n_size * 2, n_size * 2
    return K, kernel_size






def gauss_rff_select_lmskb(X, Y, p = 2 ** 10, one_sided=False, n_sigma=20):
    
    # X = n_size, n_image_vectorize * 2
    # Y = n_size, n_image_vectorize * 2
    
    def rff(X, sigma):
        # X = n_size, d; d= 784 * 2
        # sigma = scalar
        # omegas = d, p
        # thetas = p, 1
        return np.sqrt(2 / p) * np.cos(1 / sigma * (X @ omegas) + thetas.T) # n_size, p

    def rff2(X, sigma):
        return rff(X, sigma / np.sqrt(2)) # n_size, p

    # to approximate kernel size for deflation

    d = X.shape[1]
    thetas = 2 * np.pi * np.random.rand(p, 1) # p, 1
    # omegas = np.random.rand(d, p)  # d, p
    # omegas = norm.ppf(np.random.rand(d,p))
    omegas = np.random.randn(d,p)
    
    X_center = X - np.mean(X, axis = 0, keepdims=True) # n_size, n_image_vectorize * 2
    Y_center = Y - np.mean(Y, axis = 0, keepdims=True) # n_size, n_image_vectorize * 2
    median_norm_from_means = np.median(np.sqrt(np.sum(np.concatenate([X_center, Y_center], axis=1) **2, axis=0, keepdims = True)), axis=1, keepdims = True) # 1, 1

    # sigmas = (median_norm_from_means.squeeze() * np.logspace(-3, 3, n_sigma))# [np.newaxis, ...] # 1, n_sigma
    sigmas = (median_norm_from_means.squeeze() * np.logspace(-2, 2, n_sigma))# [np.newaxis, ...] # 1, n_sigma

    # div = np.nan((np.size(sigmas), 1))
    div = np.full((np.size(sigmas)), np.nan) # n_sigma

    for sigma_i in range(np.size(sigmas)):
        sigma = sigmas[sigma_i] # scalar

        Phi_XT = rff2(X, sigma) # n_size, p
        Phi_YT = rff2(Y, sigma) # n_size, p
        mu_X = np.mean(Phi_XT, axis=0, keepdims=True).T #p, 1
        mu_Y = np.mean(Phi_YT, axis=0, keepdims=True).T #p, 1

        landmark_sliced_kernel_bures = np.sqrt(np.maximum(0, Phi_XT @ mu_X)) - np.sqrt(np.maximum(0, Phi_YT @ mu_Y)) # n_size, 1
        if one_sided:
            div[sigma_i] = np.max(landmark_sliced_kernel_bures) # scalar
        else:
            div[sigma_i] = np.max(np.abs(landmark_sliced_kernel_bures)) # scalar

    sigma_star = sigmas[np.argmax(div)] # scala
    rff_out = rff(X, sigma_star)

    info = None #np.ones(np.product(sigmas.shape), 2)
    # for sigma_i in range(np.product(sigmas.shape)):
    #     sigma = sigmas[sigma_i]
    #     info[sigma_i, 0] = bures_lb_embed_info(rff(X, sigma).T)
    #     info[sigma_i, 1] = bures_lb_embed_info(rff(Y, sigma).T)

    return rff_out, sigmas, div, info
