clc;
close all;
clear all;

% xx = readmatrix('DT.csv');
%% 生成高斯分布数据
figure(1);
mu = [1 1];
SIGMA = [0.1 0; 0 0.1];
r = mvnrnd(mu,SIGMA,300);
plot(r(:,1),r(:,2),'r+');
hold on;
mu = [1 5];
SIGMA = [ 0.1 0; 0 0.1];
r2 = mvnrnd(mu,SIGMA,400);
plot(r2(:,1),r2(:,2),'*');
hold on;
mu = [5 5];
SIGMA = [ 0.1 0; 0 0.1];
r3 = mvnrnd(mu,SIGMA,300);
plot(r3(:,1),r3(:,2),'.')

a = zeros(300,1);
b = zeros(400,1);
r = [r a+1];
r2 = [r2 b+2];
r3 = [r3 a+3];
xx = [r;r2;r3];

rand('seed',10);
[n,d] = size(xx);
x = xx(:,1:d-1);
[n,d] = size(x);
true_class = xx(:,d+1);  %%% class label

x_numeric = x; %%%% numerical part of the data
k = max(true_class);
k = 6;
[n, d] = size(x);  %%% n: the number of samples;   d: the number of attributes(dimensionality)
[~, d_n] = size(x_numeric);


%% Normalization to the numerical part %%%%%%%%%%%%%%%%%
% for i=1:d_n
%     x_numeric(:,i) = (x_numeric(:,i)-mean(x_numeric(:,i)))./std(x_numeric(:,i));
% end

%% Implementation of RPCCL algorithm %%%%%
T_total = 200; %%% Execute 100 times to get the average performance
epoch_record = zeros(1, T_total);
error_record = zeros(1, T_total);
time_record = zeros(1, T_total);
Z = zeros(1,T_total);
%% Randomly initialize the k seed points(one for each class) %%%%%
%%%%initialization
miu_numer = zeros(k, d_n);
%随机设置一个样本为cluster
ran = randperm(n);
for i = 1:k
    miu_numer(i,:) = x_numeric(ran(i), :);
end

miu_numer

win_count = ones(1, k); %%% winning times of each cluster
for T_times = 1:T_total
    tic
    cluster_label = zeros(n, k); %%% cluster label of each object

    I = zeros(1,k);
    a_c = 0.001;     %%delearning rate
    count_miu = cell(1,n);

    ran_object = randperm(n);
    for i = 1:n
        %% 计算class label
        class_distance = zeros(1,k);
        for l = 1:k
            count_dis = x(i,:) - miu_numer(l,:);
            class_distance(l) =  sum(count_dis .* count_dis);
        end
        c = find(class_distance == min(class_distance));
        win_count(c) = win_count(c) + 1;    %%更新win count
        cluster_label(i,c) = 1;
        %% 计算I find rival
        I = zeros(n,k);
        I_distance = zeros(1,k);
        for l = 1:k
            count_dis = x(i,:) - miu_numer(l,:);
            I_distance(l) = win_count(l)/sum(win_count) * sum(count_dis .* count_dis);
        end
        j = find(I_distance == min(I_distance));

%         if(j == c)
%             I(i,j) = 1;
%         else
%             I(i,j) = -1;
%         end
        %find the rival
        if(j == c)
            b=sort(I_distance);
            r = find(b(2) == I_distance);
        else
            r = j;
        end
        if r == c
            fault = 1;
        end
        %% update winner and rival with penalization control
%         rival_distance = zeros(1,k);
%         for l = 1:k
%             count_dis = miu_numer(c,:) - miu_numer(l,:);
%             rival_distance(l) = sum(count_dis .* count_dis);
%         end
%         b=sort(rival_distance);
%         r = find(b(2) == rival_distance);

        dis_r_c = sqrt(sum((miu_numer(r,:) - miu_numer(c,:)) .* (miu_numer(r,:) - miu_numer(c,:))));
        dis_c_x = sqrt(sum((miu_numer(c,:) - x(i,:)) .* (miu_numer(c,:) - x(i,:))));
        pen_control = min(miu_numer(r,:) - miu_numer(c,:));
        % 更新winer m_c和rival m_r
        miu_numer(c,:) = miu_numer(c,:) + a_c * (x(i,:) - miu_numer(c,:));
        miu_numer(r,:) = miu_numer(r,:) - a_c * (min(dis_r_c,dis_c_x)/dis_r_c) * (x(i,:) - miu_numer(r,:));
        miu_numer;
        count_miu{i} = miu_numer;
    end
    z = 0;
    for l = 1:k
        for i = 1:n
            c = find(cluster_label(i,:) == 1);
            z = z + sqrt(sum((miu_numer(c,:) - x(i,:)) .* (miu_numer(c,:) - x(i,:))));
        end
    end
    z;
    Z(T_times) = z;
    if(T_times >=2)
        z_change = abs(z - Z(T_times-1))
    end
    miu_numer;
end

miu_numer
figure(2);
mark = ['+', '*', '.', '>', '^', 'o'];
for i = 1:n
    label = find(cluster_label(i,:) == 1);
    plot(x(i,1),x(i,2),mark(label),"Color",'black');
    hold on;
end

