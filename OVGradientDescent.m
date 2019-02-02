%% OVGradientDescent.m
% 单变量线性回归
% 使用梯度下降技术，线性拟合奥运会自由泳100米记录

% The Elements of Machine Learning ---- Principles Algorithms and Practices
% Author Mike Yuan, Copyright 2016~2017

%% 主函数
function OVGradientDescent()

%% 初始化
clear; close all; clc


%% 加载数据
load('../data/Freestyle100m');
x = men100(:, 1);
y = men100(:, 2);
N = length(y); % 训练样本数

% 显示数据
figure;
plot(x, y, 'ro', 'MarkerSize', 6); % 显示数据点
ylabel('取胜时间（秒）');      % 设置y轴标签
xlabel('奥运会年');         % 设置x轴标签

%% 梯度下降
% 为方便数值运算，将原来的举办年减去第一届奥运会年（1896），并添加一列全1，以扩展x
x = [ones(N, 1), x - 1896];
w = zeros(2, 1);        % 参数初始值

% 梯度下降法的设置
iterations = 50000; % 迭代次数
alpha = 0.00034;	% 学习率

% 调用梯度下降函数
[w, Jhistory] = gradientDescent(x, y, w, alpha, iterations);

% 打印找到的参数
fprintf('梯度下降找到的w0和w1：%f %f \n', w(1), w(2));

% 绘制线性拟合直线
hold on;	% 保持原来的绘图可见
plot(x(:, 2) + 1896, x * w, '-');    % 绘制直线
legend('训练数据', '线性回归');
hold off;

%% 可视化 J(w0, w1)

% 代价函数J的绘图范围
w0 = linspace(-30, 170, 100);
w1 = linspace(-2, 2, 100);

% 初始化J的值
J = zeros(length(w0), length(w1));

% 填充J的值
for ii = 1 : length(w0)
    for jj = 1 : length(w1)
	  t = [w0(ii); w1(jj)];    
	  J(ii, jj) = computeCost(x, y, t);
    end
end

J = J';
% 绘制曲面图
figure;
surf(w0, w1, J);
xlabel('w0'); ylabel('w1');

% 绘制等值线图
figure;
% 绘制代价J的等值线图
contour(w0, w1, J, logspace(-2, 3, 20))
xlabel('w0'); ylabel('w1');
hold on;
plot(w(1), w(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

% 绘制代价J的下降曲线
figure;
plot(1 : length(Jhistory), Jhistory, 'r-'); % 绘制数据散点图
xlabel('迭代次数');
ylabel('代价J');

end

%% 梯度下降函数，找到合适的参数
% 输入参数
%   x：输入，y：输出，w：截距和斜率参数，alpha：学习率，iters：迭代次数
% 输出参数
%   w：学习到的截距和斜率参数，Jhistory：迭代计算的J值历史
function [w, Jhistory] = gradientDescent(x, y, w, alpha, iters)
% 初始化
N = length(y);	% 训练样本数
Jhistory = zeros(iters, 1);

for iter = 1 : iters
    % 要求同时更新w参数，因此设置两个临时变量temp0和temp1
    temp0 = w(1) - alpha / N * (x(:, 1)' * (x * w - y));
    temp1 = w(2) - alpha / N * (x(:, 2)' * (x * w - y));
    w(1) = temp0;
    w(2) = temp1;

    % 保存代价J
    Jhistory(iter) = computeCost(x, y, w);
end

end

%% 计算线性回归的代价
% 使用w作为线性回归的参数，计算代价J
% 输入参数
%	x：输入，y：输出，w：截距和斜率参数
% 输出参数
%	J：计算的J值
function J = computeCost(x, y, w)
N = length(y);	% 训练样本数

J = 1 / (2 * N) * sum((x * w - y) .^ 2);

end


