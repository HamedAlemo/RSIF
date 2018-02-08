function [data1, data2] = removeNaN(data1, data2)
% This function removes NaN values from both datasets that are input and
% target in NN

[~, iNAN] = find(isnan(data1));
data1(:, iNAN) = [];
data2(:, iNAN) = [];

[~, iNAN] = find(isnan(data2));
data1(:, iNAN) = [];
data2(:, iNAN) = [];
