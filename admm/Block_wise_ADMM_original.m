function [out] = Block_wise_ADMM_original(A, b, opts, x_true)
%% Main Function_Direct Extension of ADMM

%% Parameters Initialization
x = zeros(opts.Group_number,1);
x_0 = zeros(opts.Group_number,1);
lambda_0 = zeros(opts.Block_row,1);

%% Output Initialization
out.obj = zeros(opts.Max_iter,1);
out.dist = zeros(opts.Max_iter,1);
out.dist = zeros(opts.Max_iter,1);
out.constraints = zeros(opts.Max_iter,1);

%% Pre_Computing
term_group = zeros(opts.Block_row,1);
for i = 1:opts.Group_number
    term_group = term_group + A{i}*x_0(i);
end;
out.time_sum = 0;

%% Main_iteration
for iter = 1:opts.Max_iter
    
    t_0 = cputime;
    %% Block_wise_ADMM_Prediction_Step_(ADMM + Parallel Computing)
    for i = 1:opts.Group_number
        term_a = b + lambda_0/opts.Beta - term_group + A{i}*x_0(i);
       
       %% Shrinkage_operator
        term_x = (term_a'*A{i})/(A{i}'*A{i});
        x(i) = term_x - max(min( term_x, 1/(opts.Beta*(A{i}'*A{i}))), - 1/(opts.Beta*(A{i}'*A{i})));
       
       %% Compute the temporary term    
        term_group = term_group + A{i}*( x(i) - x_0(i) );
        
    end;
    
    %% Update $\lambda$
    lambda = lambda_0 - opts.Beta*(term_group - b);
    out.time(iter) = cputime - t_0;
%     %%
%     term_Group = A{opts.Group_number}*(x(opts.Group_number) - x_0(opts.Group_number));
%     for t = ( opts.Group_number - 1):(-1):2
%         x(t) = x(t) - (A{t}'*term_Group)/((tau + 1)*A{t}'*A{t});
%         term_Group = term_Group + A{t}*(x(t) - x_0(t));
%     end;
    out.dist(iter) = norm(x - x_0, 2);    
    %% Update new variables
    lambda_0 = lambda;  x_0 = x;    % Tansfer_new_variables
    out.time_sum = out.time_sum + out.time(iter);
    %% Computer the output information
    out.obj(iter) = norm(x, 1);  % Objective_function_value
    out.constraints(iter) = norm(term_group - b, 2);    % Constriants_violation_value
    out.variable = x;
    out.iter = iter;
    out.error = out.dist(iter);
    if ( out.error < 1e-4 )
        fprintf('ADMM-Direct ,iteration = %5d , time=%5f, error = %5d\n', out.iter, out.time_sum, out.error);
        break;
    end;
%     fprintf('iteration = %d\n',iter);
end;

end

