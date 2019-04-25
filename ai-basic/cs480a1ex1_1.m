%cs480a1ex1_1

%cleanup
clear all;
close all;

%load
load spambase_X.mat;
load spambase_Y.mat;

%init
d = 57;
n = 4601;
w = zeros([d 1]);
b = 0;
max_pass = 500;

%Algo 1
for t = 1:max_pass
    mistake(t) = 0;
    for i = 1:n
        if (y(i,:) .* (dot(X(:,i),w) + b)) <= 0
            w = w + y(i,:) .* X(:,i);
            b = b + y(i,:);
            mistake(t) = mistake(t) + 1;
        end
    end
end

%plot
scatter(1:500, mistake)