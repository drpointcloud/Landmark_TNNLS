# get from https://github.com/steverab/failing-loudly
# may/might have been modified
# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

from cmath import nan
import numpy as np
from numpy.lib.function_base import median
from tensorflow.python.keras.backend import flatten, reshape
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.training.tracking import data_structures
import torch
import random
from torch.autograd import variable
# from torch import *
from torch_two_sample import *
from scipy.stats import ks_2samp, binom_test, chisquare, chi2_contingency, anderson_ksamp
from scipy.spatial import distance

from scipy import sparse
from shared_utils import *


# -------------------------------------------------
# SHIFT TESTER
# -------------------------------------------------

class ShiftTester:

    def __init__(self, dim=TestDimensionality.One, sign_level=0.05, ot=None, mt=None):
        self.dim = dim
        self.sign_level = sign_level
        self.ot = ot
        self.mt = mt

    def test_shift(self, X_tr, X_te):
        if self.ot is not None:
            return self.one_dimensional_test(X_tr, X_te)
        elif self.mt is not None:
            return self.multi_dimensional_test(X_tr, X_te)

    def test_chi2_shift(self, X_tr, X_te, nb_classes):

        # Calculate observed and expected counts
        freq_exp = np.zeros(nb_classes)
        freq_obs = np.zeros(nb_classes)

        unique_tr, counts_tr = np.unique(X_tr, return_counts=True)
        total_counts_tr = np.sum(counts_tr)
        unique_te, counts_te = np.unique(X_te, return_counts=True)
        total_counts_te = np.sum(counts_te)

        for i in range(len(unique_tr)):
            val = counts_tr[i]
            freq_exp[unique_tr[i]] = val
            
        for i in range(len(unique_te)):
            freq_obs[unique_te[i]] = counts_te[i]

        if np.amin(freq_exp) == 0 or np.amin(freq_obs) == 0:
            # The chi-squared test using contingency tables is not well defined if zero-element classes exist, which
            # might happen in the low-sample regime. In this case, we calculate the standard chi-squared test.
            #for i in range(len(unique_tr)):
            #    val = counts_tr[i] / total_counts_tr * total_counts_te
            #    freq_exp[unique_tr[i]] = val
            #_, p_val = chisquare(freq_obs, f_exp=freq_exp)
            p_val = random.uniform(0, 1)
        else:
            # In almost all cases, we resort to obtaining a p-value from the chi-squared test's contingency table.
            freq_conc = np.array([freq_exp, freq_obs])
            _, p_val, _, _ = chi2_contingency(freq_conc)
        
        return p_val

    def test_shift_bin(self, k, n, test_rate):
        p_val = binom_test(k, n, test_rate)
        return p_val

    def one_dimensional_test(self, X_tr, X_te):
        p_vals = []

        # For each dimension we conduct a separate KS test
        #for i in range(X_tr.shape[1]):
        for i in range(1):
            feature_tr = X_tr#[:, i]
            feature_te = X_te#[:, i]

            t_val, p_val = None, None


            if self.ot == OnedimensionalTest.KS:

                # Compute KS statistic and p-value
                t_val, p_val = ks_2samp(feature_tr, feature_te)
                #print("first possible: ", p_val)
            elif self.ot == OnedimensionalTest.AD:
                t_val, _, p_val = anderson_ksamp([feature_tr.tolist(), feature_te.tolist()])
                #print("second possible: ", p_val)


            # added ##################################################################>
            elif self.ot == OnedimensionalTest.LMSW:
                Z = np.atleast_2d(np.transpose(np.concatenate((feature_tr,feature_te),axis=0))) # combine two samples
                x_idx = np.concatenate((np.ones(feature_tr.shape[0]),np.zeros(feature_te.shape[0])),axis=0).astype(np.bool) #bool indicator
                #np.array([1, 0, 1, 0]).astype(np.bool)
                K,sigma = self.gaussian_kernel(np.transpose(Z))
                m = np.sum(x_idx)
                n = K.shape[0] - m # sample size for Y
                K = (K + np.transpose(K))/2 #  ensure symmetric  (doesn't ensure PSD)
                K_X_Z  = K[ x_idx, :]
                K_Y_Z  = K[~x_idx, :]

                if m==n:  # assumes the X and Y are equal size 
                    landmark_divs = np.mean(np.square(np.sort(K_X_Z,axis=0) - np.sort(K_Y_Z,axis=0)) ,axis=0); 
                else: #if m is not equal to n, mass spliting applied for exact corresponding
                    m_part = self.linear_space(0,1,1/m) #lispace(start,stop,step)
                    n_part = self.linear_space(0,1,1/n)
                    G = np.minimum(np.atleast_2d(m_part).T,np.atleast_2d(n_part))
                    P = np.diff(np.diff(G,axis=0),axis=1)  #matlab: diff(data,nth order,dimention)
                    landmark_divs = np.mean(np.square(K_X_Z),axis=0)+ np.mean(np.square(K_Y_Z),axis=0)- 2*np.sum(np.multiply((np.transpose(P)@np.sort(K_X_Z,axis=0)),np.sort(K_Y_Z,axis=0)) ,axis=0)

                idx = np.unravel_index(np.argmax(landmark_divs, axis=None), landmark_divs.shape)
                max_val = np.nanmax(landmark_divs)
                div_max = np.sqrt(np.maximum(0,max_val))
                #div_mean = mean(landmark_divs,'omitnan')
                #divs = div_max;#%[div_max div_mean]
                alphas = np.zeros(np.size(K,axis=0))
                alphas[idx] = 1
                V = K@alphas
                #return div_max

                
                tests = 1000
                if (tests>1):
                    shuffled_tests = np.zeros((tests,1))
                    K_tilde = np.sort(K,axis=0)
                    if m==n:
                        for t in range(tests):
                            rand_perm = x_idx[np.random.permutation(len(x_idx))] #bool
                            #p2(t,:) = sqrt(max(0,max(mean((K_tilde(rand_perm,:) - K_tilde(~rand_perm,:)).^2,1))));
                            shuffled_tests[t] = np.sqrt(np.maximum(0,np.max(np.mean(np.square(K_tilde[rand_perm,:] - K_tilde[~rand_perm,:]),axis=0))))
                    else:
                        for t in range(tests):
                            rand_perm = x_idx[np.random.permutation(len(x_idx))]
                            KXZ = K_tilde[rand_perm,:]
                            KYZ = K_tilde[~rand_perm,:]
                            #sqrt(max(0,max(mean(KXZ.^2,1) + mean(KYZ.^2,1) - 2*sum((P'*KXZ).*KYZ,1))));
                            shuffled_tests[t] = np.sqrt(np.maximum(0,np.maximum(np.mean(np.square(KXZ),axis=0) + np.mean(np.square(KYZ),axis=0) - 2*np.sum((np.transpose(P)*KXZ)@KYZ,axis=0) )))
                    
                p_val = np.mean(shuffled_tests>=div_max)
                #print("third possible: ", p_val)




            p_vals.append(p_val)

        # Apply the Bonferroni correction to bound the family-wise error rate. This can be done by picking the minimum
        # p-value from all individual tests.
        p_vals = np.array(p_vals)
        #print(p_vals)
        p_val = min(np.min(p_vals), 1.0)

        return p_val, p_vals, landmark_divs, V

    def multi_dimensional_test(self, X_tr, X_te):

        # torch_two_sample somehow wants the inputs to be explicitly casted to float 32.
        X_tr = X_tr.astype(np.float32)
        X_te = X_te.astype(np.float32)

        p_val = None

        # We provide a couple of different tests, although we only report results for MMD in the paper.
        if self.mt == MultidimensionalTest.MMD:
            mmd_test = MMDStatistic(len(X_tr), len(X_te))

            # original code ################################################################
            # # As per the original MMD paper, the median distance between all points in the aggregate sample from both
            # # distributions is a good heuristic for the kernel bandwidth, which is why compute this distance here.
            # if len(X_tr.shape) == 1:
            #     X_tr = X_tr.reshape((len(X_tr),1))
            #     X_te = X_te.reshape((len(X_te),1))
            #     all_dist = distance.cdist(X_tr, X_te, 'euclidean')
            # else:
            #     all_dist = distance.cdist(X_tr, X_te, 'euclidean')
            # median_dist = np.median(all_dist)

            # # Calculate MMD.
            # t_val, matrix = mmd_test(torch.autograd.Variable(torch.tensor(X_tr)),
            #                          torch.autograd.Variable(torch.tensor(X_te)),
            #                          alphas=[1/median_dist], ret_matrix=True)
            # p_val = mmd_test.pval(matrix)
            ################################################################

            # The correct median kernel size calculation below
            Z = (np.concatenate((X_tr,X_te),axis=0)) # combine two samples
            _,median_dist = gaussian_kernel(Z)
            # Calculate MMD.
            t_val, matrix = mmd_test(torch.autograd.Variable(torch.tensor(X_tr)),
                                     torch.autograd.Variable(torch.tensor(X_te)),
                                     alphas=[1/(2*median_dist**2)], ret_matrix=True)
            p_val = mmd_test.pval(matrix)
            
        elif self.mt == MultidimensionalTest.Energy:
            energy_test = EnergyStatistic(len(X_tr), len(X_te))
            t_val, matrix = energy_test(torch.autograd.Variable(torch.tensor(X_tr)),
                                        torch.autograd.Variable(torch.tensor(X_te)),
                                        ret_matrix=True)
            p_val = energy_test.pval(matrix)

        elif self.mt == MultidimensionalTest.FR:
            fr_test = FRStatistic(len(X_tr), len(X_te))
            t_val, matrix = fr_test(torch.autograd.Variable(torch.tensor(X_tr)),
                                    torch.autograd.Variable(torch.tensor(X_te)),
                                    norm=2, ret_matrix=True)
            p_val = fr_test.pval(matrix)

        elif self.mt == MultidimensionalTest.KNN:
            knn_test = KNNStatistic(len(X_tr), len(X_te), 20)
            t_val, matrix = knn_test(torch.autograd.Variable(torch.tensor(X_tr)),
                                     torch.autograd.Variable(torch.tensor(X_te)),
                                     norm=2, ret_matrix=True)
            p_val = knn_test.pval(matrix)


        # added ##################################################################>
        elif self.mt == MultidimensionalTest.LMSW:
                
            Z = (np.concatenate((X_tr,X_te),axis=0)) # combine two samples
            x_idx = np.concatenate((np.ones(X_tr.shape[0]),np.zeros(X_te.shape[0])),axis=0).astype(np.bool) #bool indicator
            #np.array([1, 0, 1, 0]).astype(np.bool)

            K,sigma = self.gaussian_kernel(Z)
            m = np.sum(x_idx)
            n = K.shape[0] - m # sample size for Y

            K = (K + np.transpose(K))/2 #  ensure symmetric  (doesn't ensure PSD)
            #K  = K*diag(diag(K).^(-1)); % normalize
            #K  = K*np.diagonal(np.diagonal(K)**-1) # normalize
            K_X_Z  = K[ x_idx, :]
            K_Y_Z  = K[~x_idx, :]

            if m==n:  # assumes the X and Y are equal size
                #landmark_divs = mean( (sort(K_X_Z) - sort(K_Y_Z)).^2 , 1);    
                landmark_divs = np.mean(np.square(np.sort(K_X_Z,axis=0) - np.sort(K_Y_Z,axis=0)) ,axis=0); 
            else: #if m is not equal to n, mass spliting applied for exact corresponding
                m_part = self.linear_space(0,1,1/m) #lispace(start,stop,step)
                n_part = self.linear_space(0,1,1/n)
                G = np.minimum(np.atleast_2d(m_part).T,np.atleast_2d(n_part))
                P = np.diff(np.diff(G,axis=0),axis=1)  #matlab: diff(data,nth order,dimention)
                #landmark_divs = mean(K_X_Z.^2,1) + mean(K_Y_Z.^2,1) - 2*sum((P'*sort(K_X_Z)).*sort(K_Y_Z) ,1);
                landmark_divs = np.mean(np.square(K_X_Z),axis=0)+ np.mean(np.square(K_Y_Z),axis=0)- 2*np.sum(np.multiply((np.transpose(P)@np.sort(K_X_Z,axis=0)),np.sort(K_Y_Z,axis=0)) ,axis=0)

            idx = np.unravel_index(np.argmax(landmark_divs, axis=None), landmark_divs.shape)
            max_val = np.nanmax(landmark_divs)
            div_max = np.sqrt(np.maximum(0,max_val))
            #div_mean = mean(landmark_divs,'omitnan')
            #divs = div_max;#%[div_max div_mean]
            alphas = np.zeros(np.size(K,axis=0))
            alphas[idx] = 1
            V = K@alphas
            #return div_max
            
            tests = 1000
            if (tests>1):
                shuffled_tests = np.zeros((tests,1))
                K_tilde = np.sort(K,axis=0)
                if m==n:
                    for t in range(tests):
                        rand_perm = x_idx[np.random.permutation(len(x_idx))] #bool
                        #p2(t,:) = sqrt(max(0,max(mean((K_tilde(rand_perm,:) - K_tilde(~rand_perm,:)).^2,1))));
                        shuffled_tests[t] = np.sqrt(np.maximum(0,np.max(np.mean(np.square(K_tilde[rand_perm,:] - K_tilde[~rand_perm,:]),axis=0))))
                else:
                    for t in range(tests):
                        rand_perm = x_idx[np.random.permutation(len(x_idx))]
                        KXZ = K_tilde[rand_perm,:]
                        KYZ = K_tilde[~rand_perm,:]
                        #sqrt(max(0,max(mean(KXZ.^2,1) + mean(KYZ.^2,1) - 2*sum((P'*KXZ).*KYZ,1))));
                        shuffled_tests[t] = np.sqrt(np.maximum([0],np.max(np.mean(np.square(KXZ),axis=0) + np.mean(np.square(KYZ),axis=0) - 2*np.sum((np.transpose(P)*KXZ)@KYZ,axis=0) )))
                
            p_val = np.mean(shuffled_tests>=div_max)
            p_vals = np.array(p_val)
    
        return p_val, None , landmark_divs, V


    # added ##################################################################
    def gaussian_kernel(self, X):# assumes the sigma is median 
        n,d = X.shape # [n,d] = size(X);# Assume Gaussian kernel
        # sigma2 = np.logspace(-4,4,30) # %sigma2s = logspace(-4,4,30);
        # D2 = max(0,  -2*(X*X.') + sum(X.^2,2) + sum(X.^2,2).'); # % Rely on squared Euclidean distances
        D2 = np.maximum(0,-2*(X@np.transpose(X)) + np.sum(np.square(X),axis=1) + np.transpose(np.sum(np.square(X),axis=1))) 
        sigma = np.nanmedian(np.ravel(np.sqrt(D2)+ sparse.spdiags(nan,0,n,n)))

        if sigma.size == 1:
            K = np.exp(-D2/(2*sigma**2))
        else:
            X = X*sparse.spdiags(sigma,0,d,d)
            # Rely on squared Euclidean distances
            D2 = np.maximum(np.zeros((n,n)),-2*(X*np.transpose(X)) + np.sum(np.square(X),axis=1) + np.transpose(np.sum(np.square(X),axis=1)))
            K = np.exp(-D2/(2*sigma**2))

        K  = (K+np.transpose(K))/2
        return K,sigma

    def linear_space(self,start, stop, step=1.):
        """
        Like np.linspace but uses step instead of num
        This is inclusive to stop, so if start=1, stop=3, step=0.5
        Output is: array([1., 1.5, 2., 2.5, 3.])
        """
        return np.linspace(start, stop, int((stop - start) / step + 1))
        # added ##################################################################
            
        


