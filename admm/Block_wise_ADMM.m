function [out] = Block_wise_ADMM(A, b, opts, x_true)
%% Main Function_Block_wise_ADMM_with_Gaussian_back_substitution

%% Re_grouping
if opts.Regroup_number > opts.Group_number
    fprintf('Regroup block number is smaller than original block number~');
end;
opts.Sub_block_number = opts.Group_number/opts.Regroup_number; % we can also set opts.Sub_block_number as a vector

%% Initialization
x = zeros(opts.Group_number,1);
x_term = zeros(opts.Group_number,1);
x_0 = zeros(opts.Group_number,1);
lambda_0 = zeros(opts.Block_row,1);

% AiA = cell(opts.Regroup_number,1);
% for i = 1:opts.Regroup_number
%     for j = 1:opts.Sub_block_number
%         AiA{(i-1)*opts.Sub_block_number + j} = inv(A{(i-1)*opts.Sub_block_number + j}'*A{(i-1)*opts.Sub_block_number + j});
%     end;
% end;

%% Generate the Block-wise $A$
A_Regroup = cell(opts.Regroup_number,1);
for i = 1:opts.Regroup_number
    A_Regroup{i} = zeros(opts.Block_row, opts.Sub_block_number);
end;
for i = 1:opts.Regroup_number
    for j = 1:opts.Sub_block_number
        A_Regroup{i}(:,j) = A{((i - 1)*opts.Sub_block_number + j)};
    end;
end;

%% Output Initialization
out.obj = zeros(opts.Max_iter,1);
out.dist = zeros(opts.Max_iter,1);
out.error = zeros(opts.Max_iter,1);
out.constraints = zeros(opts.Max_iter,1);

%% Pre_Computing
term_group = zeros(opts.Block_row,1);
for i = 1:opts.Group_number
    term_group = term_group + A{i}*x_0(i);
end;
out.time_term = zeros(opts.Sub_block_number,1);
out.time_sum = 0;

%% Main_iteration
for iter = 1:opts.Max_iter
    
    t_0 = cputime;
    tau = opts.Sub_block_number - 1;
    %% Block_wise_ADMM_Prediction_Step_(ADMM + Parallel Computing)
    for i = 1:opts.Regroup_number
        t_00 = cputime;
        for j = 1:opts.Sub_block_number
            term_a = b + lambda_0/opts.Beta - term_group + A{(i-1)*opts.Sub_block_number + j}*x_0((i-1)*opts.Sub_block_number + j);
            term_b = A{(i-1)*opts.Sub_block_number + j}*x_0((i-1)*opts.Sub_block_number + j);
          %% Shrinkage_operator
            term_x = (A{(i-1)*opts.Sub_block_number + j}'*term_a + tau*A{(i-1)*opts.Sub_block_number + j}'*term_b)/((tau + 1)*A{(i-1)*opts.Sub_block_number + j}'*A{(i-1)*opts.Sub_block_number + j});
            x_term((i-1)*opts.Sub_block_number + j) = term_x - max(min( term_x, 1/((tau + 1)*opts.Beta*A{(i-1)*opts.Sub_block_number + ...
                j}'*A{(i-1)*opts.Sub_block_number + j})), - 1/((tau + 1)*opts.Beta*A{(i-1)*opts.Sub_block_number + j}'*A{(i-1)*opts.Sub_block_number + j}));
        end;
        out.time_term(i) = cputime - t_00;
        for j = 1:opts.Sub_block_number
            term_group = term_group + A{(i-1)*opts.Sub_block_number + j}*(x_term((i-1)*opts.Sub_block_number + j) - x_0((i-1)*opts.Sub_block_number + j));
        end;
    end;
    
    %% Update $\lambda$
    lambda = lambda_0 - opts.alpha*opts.Beta*(term_group - b);
    out.time(iter) = cputime - t_0 - (opts.Sub_block_number - 1)*(sum(out.time_term))/(opts.Sub_block_number);
    x = x_term;
    out.dist(iter) = norm(x - x_0, 2);
    %% Gaussian_back_substitution
%     term_GBS_sum = zeros(opts.Block_row,1);
%     for j = 1:opts.Sub_block_number
%         x(opts.Group_number + 1 - j) = x_0(opts.Group_number + 1 - j) + opts.alpha*( x_term(opts.Group_number + 1 - j) - x_0(opts.Group_number + 1 - j) );
%     end;
%     for i = (opts.Regroup_number - 1):(-1):2
% %         term_GBS_sum = zeros(opts.Block_row,1);
% %         for t = (i*opts.Sub_block_number + 1):opts.Group_number
% %             term_GBS_sum = term_GBS_sum + A{t}*( x(t) - x_0(t) );
% %         end;
%         for j = 1:opts.Sub_block_number
%             x((i-1)*opts.Sub_block_number + j) = x_0((i-1)*opts.Sub_block_number + j) +  opts.alpha*( x_term((i-1)*opts.Sub_block_number + j) - x_0((i-1)*opts.Sub_block_number + j) ) - ...
%                 ( A{(i-1)*opts.Sub_block_number + j}'*term_GBS_sum )/( ( tau + 1 )*(A{(i-1)*opts.Sub_block_number + j}'*A{(i-1)*opts.Sub_block_number + j}) );
%         end;
%         for j = 1:opts.Sub_block_number
%             term_GBS_sum = term_GBS_sum + A{(i-1)*opts.Sub_block_number + j}*( x((i-1)*opts.Sub_block_number + j) - x_0((i-1)*opts.Sub_block_number + j));
%         end;
%     end;
%     for j = 1:opts.Sub_block_number
%         x(j) = x_0(j) + opts.alpha*( x_term(j) - x_0(j) );
%     end;
    
    %% Update new variables
    lambda_0 = lambda;  x_0 = x;    % Tansfer_new_variables
    out.time_sum = out.time_sum + out.time(iter);
    %% Computer the output information
    out.obj(iter) = norm(x, 1);  % Objective_function_value
%         out.error(iter) = out.error(iter) + norm(x{j} - x_0{j}, 2)^2;
    out.constraints(iter) = norm(term_group - b, 2)^2;    % Constriants_violation_value
    out.variable = x;
    out.iter = iter;
    out.error = out.dist(iter);
    if ( max( out.error/norm(x,2), out.constraints(iter)) < 1e-4 )
%         fprintf('ADMM-GBS-%d ,iteration = %5d , obj = %5f, time=%5f,  error = %5d, constraints = %5d\n', opts.Regroup_number, out.iter, out.obj(iter), out.time_sum, out.error, out.constraints(iter));
        fprintf('ADMM-GBS-%d & %5d & %5f & %5f & %5d & %5d \\\\ \n', opts.Regroup_number, out.iter, out.obj(iter), out.time_sum, out.error, out.constraints(iter));
        break;
    end;
end;
% 	fprintf('ADMM-GBS-%d & %5d & %5f & %5f & %5d & %5d \\\\ \n', opts.Regroup_number, out.iter, out.obj(iter), out.time_sum, out.error, out.constraints(iter));
end

