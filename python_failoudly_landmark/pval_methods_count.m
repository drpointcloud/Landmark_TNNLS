



% clc
% clear
% clf



%% table MLW and MMD side by side groupped by perturbation

%mnist_0.05_small_gn_shift_max_landmark
shift_size = ["small","medium","large"];
sign_level = "0.1";

methods = {'LMSW','MMD'}; %
dataset = "mnist"; %"cifar10"; %
plot_shift = "image_shift"; %'gn_shift'; %

p1 = [];
for data_ii = 1%:length(dataset)
    for sign_ii = 1%:length(sign_level)
        for shift_ii = 1 : length(shift_size)

            data_table2 =[];
            for mm_i=1:2
                method = methods{mm_i};

                % for method_i = 1:length(methods)
                if strcmp(method,'MMD')
                    fname = "%s_%s_%s_%s_MMD_pval.csv";
                else
                    fname = "%s_%s_%s_%s_max_landmark_pval.csv";
                end

                shift_name = sprintf(fname,dataset(data_ii),sign_level(sign_ii),shift_size(shift_ii),plot_shift);
                data_table = readmatrix(shift_name);
%                 data_table2 = [data_table2,data_table]

                % count tables
                n = sum(data_table<=0.1,2)   % count
                p = round(n*100/3,0); % percentage
                p1 = cat(2,p1,p)

            end
        end
    end
end

% make a latex table format for this specific example
lr_methods = ["NoRed","PCA","SRP","UAE","TAE","BBSDs"]';

d_str = string(p1);   % to sting
d_p = append(d_str,'\%'); % add character
d_p = append('&',d_p); % add character

% add character to the last column
% d_last = d_p(:,6);
d_last = append(d_p(:,6),'\\'); % add character
d_p(:,6) = d_last;

final_table = [lr_methods d_p]

%to latex
% p_latex = latex(sym(p1));
% fprintf('%s', d_p)



%% table firs MLW then MMD
% clc
% clear
% clf
% methods = {'LMSW','MMD'}; %
%
% dataset = "mnist"; %"cifar10"; %
% plot_shift = 'gn_shift'; %"image_shift"; %
% p1 = [];
% for mm_i=1:2
%     method = methods{mm_i};
%     %mnist_0.05_small_gn_shift_max_landmark
%     shift_size = ["small","medium","large"];
%     sign_level = "0.1";
%
%     % for method_i = 1:length(methods)
%     if strcmp(method,'MMD')
%         fname = "%s_%s_%s_%s_MMD_pval.csv";
%     else
%         fname = "%s_%s_%s_%s_max_landmark_pval.csv";
%     end
%
%     data_table2 =[];
%     for data_ii = 1%:length(dataset)
%         for sign_ii = 1%:length(sign_level)
%             for shift_ii = 1 : length(shift_size)
%
%                 shift_name = sprintf(fname,dataset(data_ii),sign_level(sign_ii),shift_size(shift_ii),plot_shift);
%                 data_table = readmatrix(shift_name);
%                 data_table2 = [data_table2,data_table];
%             end
%         end
%     end
%     % count tables
%     d = reshape(data_table2,6,3,3) % new data table
%     n = sum(d<=0.1,2)   % count
%     p = reshape (round(n*100/3,0),6,3) % percentage
%     p1 = cat(2,p1,p)
% end
%
% % % count tables
% % d = reshape(data_table2,6,3,3) % new data table
% % n = sum(d<=0.1,2)   % count
% % p = reshape (round(n*100/3,0),6,3) % percentage
%
% %  latex(sym(p1))
%
% % p_cell = num2cell(p1);
% % p_cell = strcat(p_cell,'\%')
% d_str = string(d1)   % to sting
% d_p = append(d_str,'\% &') % add character
% p_latex = latex(sym(p1))













