function [idx, col, val, A] = get_all_affinities(X)

n = size(X, 2);         % number of samples

% idx(i, k) represents the (k-1)-th NN of X(:, i) (include X(:, i) itself)
% D(i, k) represents the Euclidean distance between X(:, i) and X(:, idx(i, k))
[idx, D] = knnsearch(X', X', 'K', n);

% Convert Euclidean distance to affinity
% The method is refers "Zelnik-Manor, L., & Perona, P. (2004). Self-Tuning Spectral Clustering. NIPS."
% sigma_i is set to be Euclidean distance between xi and its 7-th neaest neighbor
sigma = D(:, 8);
D = exp(-(D .^ 2) ./ (sigma .* sigma(idx)));    % D(i, j) ^ 2 / (sigma(i) * sigma(idx(i, j)))

% equivalent to (idx, col, val) = find(A)
idx = sub2ind([n, n], repmat(1:n, 1, n)', idx(:));
col = repelem(1:n, 1, n)';
val = D(:);

% Normalization A
theta = sum(reshape(val .^ 2, n, n))';  % theta(k) = norm(A(:, :, k), "fro") ^ 2
val = val ./ sqrt(theta(col));

% if we need A directly
if nargout == 4, A = sparse(idx, col, val); end

end