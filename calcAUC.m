function [auc] = calcAUC(yhat,y)



auc = calcAUC_b(yhat,y);
% auc = calcAUC_c(yhat(y==1),yhat(y~=1)); %(positive,negative)


function[auc_b] = calcAUC_b(yhat,y)
    %assumes y is 0-1
    if all(or(yhat(:)==1,yhat(:)==0)) && all(or(y(:)==1,y(:)==0))
        auc_b=(sum(yhat.*y)./sum(y)+sum(1-yhat.*(y==0))./sum(y==0))/2;
    else
        if size(yhat,1)==1 
            yhat=yhat(:); % convert to a column vector
        end
        
        [cval,idx]=sort(yhat); %sort witness func. evaluations
        count_points=[diff(cval)>0;true]; % #of unique thresholds 
        y=reshape(y(idx),[],1);
        
        tp=cumsum(y); %true pos.
        fp=cumsum(1-y); %false pos.

        tpr=[0;tp(count_points)/tp(end)]; %true pos. rate
        fpr=[0;fp(count_points)/fp(end)]; %false pos. rate
        
        %apprx. of area under curve using trapezoidal rule 
        trapezoid=1/2*diff(tpr).*(fpr(1:end-1)+fpr(2:end)); 
        auc_b=sum(trapezoid);
    end
end


% 
% function [auc_c, fpr, tpr] = calcAUC_c(ypos,yneg)
%     % The positive class is the one that has lower prediction values
% 
%     yhat = cat(1, ypos, yneg);
%     y = cat(1, ones(length(ypos),1), -1*ones(length(yneg),1));
%     [~, isort] = sort(yhat);
%     y = y(isort);
%     tp = y == 1;
%     fp = ~tp;
%     tp = cumsum(tp);
%     tpr = tp/tp(end);
%     fp = cumsum(fp);
%     fpr = fp/fp(end);
%     tpr = [0; tpr];
%     fpr = [0; fpr];
%     trapezoid=1/2*diff(tpr).*(fpr(1:end-1)+fpr(2:end));
%     auc_c = 1-sum(trapezoid);
% 
% end


end




