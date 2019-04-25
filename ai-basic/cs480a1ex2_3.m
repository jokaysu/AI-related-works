%cs480a1ex2_3

%cleanup
clear all;
close all;

%load
load housing_X_test.mat;
load housing_X_train.mat;
load housing_y_test.mat;
load housing_y_train.mat;

%init
Xtrain = [Xtrain; randn(1000, 306)];
Xtest = [Xtest; randn(1000, 200)];
d=1013;
n=306;
n_valid = 31;
X_valid = Xtrain(:,1:31);
y_valid = ytrain(1:31,:);
n_train = 275;
X_train = Xtrain(:,32:306);
y_train = ytrain(32:306,:);
n_test = 200;

%ridge regression
for i = 1:11
    lambda(i) = (i-1) .* 10;
    w = (X_train * X_train.' + lambda(i) .* eye(d)) \ (X_train * y_train);
    mse_train(i) = norm(X_train.'*w - y_train)^2 / n_train;
    mse_valid(i) = norm(X_valid.'*w - y_valid)^2 / n_valid;
    mse_test(i) = norm(Xtest.'*w - ytest)^2 / n_test;
    percent(i) = nnz(w(14:1013,:)) / 1000 .* 100;
end

%output
[M, I] = min(mse_valid);
Best_lambda_star = lambda(I)
T = table(lambda.', mse_train.', mse_valid.', mse_test.', percent.');
T.Properties.VariableNames = {'Lambda_star' 'TrainMSE' 'ValidMSE' 'TestMSE' 'NonZeroPercentLast1000'};
T
