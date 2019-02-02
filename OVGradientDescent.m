%% OVGradientDescent.m
% ���������Իع�
% ʹ���ݶ��½�������������ϰ��˻�����Ӿ100�׼�¼

% The Elements of Machine Learning ---- Principles Algorithms and Practices
% Author Mike Yuan, Copyright 2016~2017

%% ������
function OVGradientDescent()

%% ��ʼ��
clear; close all; clc


%% ��������
load('../data/Freestyle100m');
x = men100(:, 1);
y = men100(:, 2);
N = length(y); % ѵ��������

% ��ʾ����
figure;
plot(x, y, 'ro', 'MarkerSize', 6); % ��ʾ���ݵ�
ylabel('ȡʤʱ�䣨�룩');      % ����y���ǩ
xlabel('���˻���');         % ����x���ǩ

%% �ݶ��½�
% Ϊ������ֵ���㣬��ԭ���ľٰ����ȥ��һ����˻��꣨1896���������һ��ȫ1������չx
x = [ones(N, 1), x - 1896];
w = zeros(2, 1);        % ������ʼֵ

% �ݶ��½���������
iterations = 50000; % ��������
alpha = 0.00034;	% ѧϰ��

% �����ݶ��½�����
[w, Jhistory] = gradientDescent(x, y, w, alpha, iterations);

% ��ӡ�ҵ��Ĳ���
fprintf('�ݶ��½��ҵ���w0��w1��%f %f \n', w(1), w(2));

% �����������ֱ��
hold on;	% ����ԭ���Ļ�ͼ�ɼ�
plot(x(:, 2) + 1896, x * w, '-');    % ����ֱ��
legend('ѵ������', '���Իع�');
hold off;

%% ���ӻ� J(w0, w1)

% ���ۺ���J�Ļ�ͼ��Χ
w0 = linspace(-30, 170, 100);
w1 = linspace(-2, 2, 100);

% ��ʼ��J��ֵ
J = zeros(length(w0), length(w1));

% ���J��ֵ
for ii = 1 : length(w0)
    for jj = 1 : length(w1)
	  t = [w0(ii); w1(jj)];    
	  J(ii, jj) = computeCost(x, y, t);
    end
end

J = J';
% ��������ͼ
figure;
surf(w0, w1, J);
xlabel('w0'); ylabel('w1');

% ���Ƶ�ֵ��ͼ
figure;
% ���ƴ���J�ĵ�ֵ��ͼ
contour(w0, w1, J, logspace(-2, 3, 20))
xlabel('w0'); ylabel('w1');
hold on;
plot(w(1), w(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

% ���ƴ���J���½�����
figure;
plot(1 : length(Jhistory), Jhistory, 'r-'); % ��������ɢ��ͼ
xlabel('��������');
ylabel('����J');

end

%% �ݶ��½��������ҵ����ʵĲ���
% �������
%   x�����룬y�������w���ؾ��б�ʲ�����alpha��ѧϰ�ʣ�iters����������
% �������
%   w��ѧϰ���Ľؾ��б�ʲ�����Jhistory�����������Jֵ��ʷ
function [w, Jhistory] = gradientDescent(x, y, w, alpha, iters)
% ��ʼ��
N = length(y);	% ѵ��������
Jhistory = zeros(iters, 1);

for iter = 1 : iters
    % Ҫ��ͬʱ����w�������������������ʱ����temp0��temp1
    temp0 = w(1) - alpha / N * (x(:, 1)' * (x * w - y));
    temp1 = w(2) - alpha / N * (x(:, 2)' * (x * w - y));
    w(1) = temp0;
    w(2) = temp1;

    % �������J
    Jhistory(iter) = computeCost(x, y, w);
end

end

%% �������Իع�Ĵ���
% ʹ��w��Ϊ���Իع�Ĳ������������J
% �������
%	x�����룬y�������w���ؾ��б�ʲ���
% �������
%	J�������Jֵ
function J = computeCost(x, y, w)
N = length(y);	% ѵ��������

J = 1 / (2 * N) * sum((x * w - y) .^ 2);

end


