function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
sum_of_data = 0;

for iter = 1:size(X, 2),
    sum_of_data = sum_of_data + theta(iter) * X(:,iter);
end;
sum_of_data = (sum_of_data - y).^2;
J = 1 / (2 * m) * sum(sum_of_data);
% =========================================================================

end
