% Intro to Artificial Neural Network (BRI516)
% Assignment 2 (by. Yejin Cho)
% 2016-04-25
clc;clear;close all;

%% (a) Solve the linear LS problem using the options (a)-(d) in Q1.
%% (a)-a. Using linear algebra
load data.txt
xi = data(:,1:3)'; % The first three columns are xi=[xi,1, xi,2, xi,3]T
yi = data(:,7)';   % and the 7th column is yi.

w = (xi*xi')\xi*yi'; % The solution for where the gradient is 0
error = 1/2*sum((w'*xi - yi).^2); % error

fprintf('The optimized w using linear algebra is [%.3f, %.3f, %.3f]\n', ...
    w(1), w(2), w(3))
fprintf('Error: %.5f\n', error);

%% (a)-b. Using gradient-based optimization
% clc; clear; close; load data.txt
% xi = data(:,1:3)'; % The first three columns are xi=[xi,1, xi,2, xi,3]T
% yi = data(:,7)';   % and the 7th column is yi.

% (1) Parameter & initial value setting
stepsize = repmat(0.01,[3,1]);  % step size (eta)
tolerance = 0.01;               % tolerance (delta; ¥ä)
w_prev = [0.5;0.5;0.5];

% (2) Define the gradient function of w (weight)
syms w1 w2 w3
weight = [w1,w2,w3];
fgradient = xi*(xi'*weight' - yi');
fgradient = matlabFunction(fgradient);  % 1st derivative (gradient)

% (3) Plot the initial figure
subplot(3,1,1)
plot(1:100, yi, 'b-')
hold on
plot(1:100, w_prev'*xi,'k*-')
title('Initial figure')

% (4) Update w in a while loop
iter = 1;

while norm(fgradient(w_prev(1), w_prev(2), w_prev(3))) > tolerance
    w_new = w_prev - stepsize .* fgradient(w_prev(1), w_prev(2), w_prev(3));
    error = 1/2*sum((w_new'*xi - yi).^2);
    fprintf('Iteration: %i\nw: [%.3f, %.3f, %.3f]\nError: %.5f\n', ...
        iter, w_new(1), w_new(2), w_new(3), error);
    
    % Plot y-hat (=xi'*w) using the optimized w vector
    subplot(3,1,2)
    plot(1:100, yi, 'b-')
    hold on
    plot(1:100, w_new'*xi,'k-*')
    hold off
    title('Optimized figure (Gradient descent)')
    
    % Plot error (squared l2 norm)
    subplot(3,1,3)
    hold on
    plot(iter, error, 'ro');
    xlabel(sprintf('Iteration: %d, w: [%.3f, %.3f, %.3f], Error: %.5f', ...
        iter, w_new(1), w_new(2), w_new(3), error));
    ylabel('Error')
    title('Error plot')
    pause(0.2)
    drawnow
    
    w_prev = w_new;     % refresh w_prev
    iter = iter + 1;    % add 1 iteration
    
end

clc;

fprintf('The optimized w using gradient descent is [%.3f, %.3f, %.3f]\n', ...
    w_new(1), w_new(2), w_new(3))
fprintf('Error: %.5f\n', error)

%% (a)-c. Using Newton's method
% clc; clear; close; load data.txt
% xi = data(:,1:3)'; % The first three columns are xi=[xi,1, xi,2, xi,3]T
% yi = data(:,7)';   % and the 7th column is yi.

% (1) Parameter & initial value setting
stepsize = repmat(0.01,[3,1]);  % step size (eta)
tolerance = 0.01;               % tolerance (delta; ¥ä)
w_prev = [0.5;0.5;0.5];

% (2) Define the gradient and hessian matrix of w (weight)
syms w1 w2 w3
weight = [w1,w2,w3];
fgradient = xi*(xi'*weight' - yi');
fgradient = matlabFunction(fgradient);  % 1st derivative of w (gradient)
hessian = xi*xi';                 % 2nd derivative of w (hessian matrix)

% (3) Plot the initial figure
figure
subplot(2,1,1)
plot(1:100, yi, 'b-')
hold on
plot(1:100, w_prev'*xi,'k*-')
title('Initial figure')

% (4) Update w deterministically using Hessian matrix
w_new = w_prev - hessian\fgradient(w_prev(1), w_prev(2), w_prev(3));
error = 1/2*sum((w_new'*xi - yi).^2);

% (5) Plot y-hat (=xi'*w) using the optimized w vector
subplot(2,1,2)
plot(1:100, yi, 'b-')
hold on
plot(1:100, xi'*w_new,'m-*')
hold off
title('Optimized figure (The Newton''s method)')

fprintf('The optimized w using the Newton''s method is [%.3f, %.3f, %.3f]\n', ...
    w_new(1), w_new(2), w_new(3))
fprintf('Error: %.5f\n', error);

%% (a)-d. Minimize Eq.(1), but subject to the constraint wTw ¡Â 1
% Namely, w'*w-1 <= 0.

f = @(w)1/2*sum((w'*xi - yi).^2);
% fmincon(f, [0,0,0], w'*w);

%% (b) Additionally, considering the degree of polynomial of xi as
% a hyperparameter of the model (i.e., candidate degrees of polynomial
% for xi from 1 to 9), solve the linear LS problem using the options
% (a)-(d) in Q1 by optimizing the hyperparameter via a nested 5-fold
% cross-validation framework.

% Note that the input dimension and the weight parameter dimension are
% variable depending on the polynomial degree.

% Results can include such as
% - training error (curves for iterative learning)
% - validation error (curves)
% - test error (curves)
% - the optimally chosen parameters during the nested cross-validation.

%% (b)-a. Using linear algebra
clc;clear;close; load data.txt
xi = data(:,1:3)'; % The first three columns are xi=[xi,1, xi,2, xi,3]T
yi = data(:,7)';   % and the 7th column is yi.

tr_error = []; val_error = [];
avg_tr_err = []; avg_val_err=[];
xtest = cell(9,1);

% Hyperparameter: the order of the polynomial (1 to 9)
for n = 1:9
    xi_n = repmat(xi, [n,1]);
    
    for degree = 1:n
        xi_n(3*(degree-1)+1:3*degree,:) = xi_n(3*(degree-1)+1:3*degree,:).^degree;
    end
    
    xtest{n} = {xi_n(:, 81:100)};
    ytest = yi(:, 81:100);
    
    % Nested 5-fold cross-validation
    for k = 1:5
        xval = xi_n(:, (k-1)*16+1:k*16);
        xtrain = xi_n(:, 1:80);
        xtrain(:, (k-1)*16+1:k*16) = [];
        
        yval = yi(:, (k-1)*16+1:k*16);
        ytrain = yi(:, 1:80);
        ytrain(:, (k-1)*16+1:k*16) = [];
        
        w = (xtrain*xtrain')\xtrain*ytrain';
        
        tr_error = [tr_error ; 1/2*sum((w'*xtrain - ytrain).^2)]; % training error
        val_error = [val_error ; 1/2*sum((w'*xval - yval).^2)]; % validation error
    end
    
    tr_error = mean(tr_error);
    val_error = mean(val_error);
    
    avg_tr_err = [avg_tr_err ; tr_error];
    avg_val_err = [avg_val_err ; val_error];
    
    w = []; % clean up w
    
end

avg_err = 1/2*(avg_tr_err + avg_val_err);
hold on
plot(avg_tr_err,'b')
plot(avg_val_err,'g')
plot(avg_err,'k')
title('(Averaged) Error plot')
xlabel('degree of polynomial')
legend('training', 'validation', 'average')

nOptimal = find(avg_err==min(avg_err));
fprintf('The least error is found in the <%ith> order polynomial model.\n',...
    nOptimal);

xtest = xtest{nOptimal}{1};
w = pinv(xtest * xtest') * xtest * ytest';
test_error = 1/2*sum((w'*xtest - ytest).^2);
fprintf('Test error: %f\n', test_error)

%% (b)-b. Using gradient-based optimization
clc;clear;close; load data.txt
xi = data(:,1:3)'; % The first three columns are xi=[xi,1, xi,2, xi,3]T
yi = data(:,7)';   % and the 7th column is yi.

% Preallocation before updating w in a while loop
iter = 1;
tr_error = []; val_error = [];
avg_tr_err = []; avg_val_err=[];
xtest = cell(9,1);

% Hyperparameter: the order of the polynomial (1 to 9)
for n = 1:9
    xi_n = repmat(xi, [n,1]);
    
    % Parameters & initial value setting
    stepsize = repmat(0.01,[3*n,1]);  % step size (eta)
    tolerance = 0.01;               % tolerance (delta; ¥ä)
    w_prev = repmat(0.5, [3*n,1]);

    for degree = 1:n
        xi_n(3*(degree-1)+1:3*degree,:) = xi_n(3*(degree-1)+1:3*degree,:).^degree;
    end
    
    xtest{n} = {xi_n(:, 81:100)};
    ytest = yi(:, 81:100);
    
    % Nested 5-fold cross-validation
    for k = 1:5
        xval = xi_n(:, (k-1)*16+1:k*16);
        xtrain = xi_n(:, 1:80);
        xtrain(:, (k-1)*16+1:k*16) = [];
        
        yval = yi(:, (k-1)*16+1:k*16);
        ytrain = yi(:, 1:80);
        ytrain(:, (k-1)*16+1:k*16) = [];
        
        fgradient = @(weight)xtrain*(xtrain'*weight'- ytrain');
        
        % Updating w in a while loop
        while norm(fgradient(w_prev')) > tolerance
            w_new = w_prev - stepsize .* fgradient(w_prev');
            tr_error = [tr_error ; 1/2*sum((w_new'*xtrain - ytrain).^2)];
            val_error = [val_error ; 1/2*sum((w_new'*xval - yval).^2)];
            
            fprintf('Iteration: %i\nError: %.5f\n', ...
                iter, 1/2*(tr_error+val_error));
            w_prev = w_new;     % refresh w_prev
            iter = iter + 1;    % add 1 iteration
        end
    end
end


