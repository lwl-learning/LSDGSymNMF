% --- More efficient parallelize version
% --- min_{X} \|X - Y \|_F^2, s.t. X >= 0, sum(X, 1) = 1
function X = proj2simplex(Y)
[n, m] = size(Y);
Z0 = Y - mean(Y, 1) + 1 / n;
Z = sort(Z0, 1);

F = (-n:-1)' .* Z > [zeros(1, m); cumsum(Z(1:end-1, :), 1)];
a = sum((F .* Z), 1) ./ (sum(F, 1) - n);
X = max(Z0 - a, 0);
end

%% Old version
% --- Close-form solution of simplex projction problem with complexity O(n log(n))
% --- min_{x} \|x - y \|_2^2, s.t. x >= 0, sum(x) = 1
% function x = proj2simplex(y)
% n = length(y);
% z0 = y - mean(y) + 1 / n;
% z = sort(z0);
% 
% f = (-n:-1)' .* z - [0; cumsum(z(1:end-1))];
% j = find(f <= 0, 1);    % f(j - 1) > 0, f(j) <= 0
% a = sum(z(1:j-1) / (j - 1 - n));
% 
% x = max(z0 - a, 0);     % note that this is z0
% end