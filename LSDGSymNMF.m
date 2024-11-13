function [V, w, p, S, D, loss, deltaVar] = LSDGSymNMF(A, V, w, p, S, D, opts, hp)
% Default algorithm options
if ~isfield(opts, "maxIter"), opts.maxIter = 1000; end
if ~isfield(opts, "tolLoss"), opts.tolLoss = 1e-4; end
if ~isfield(opts, "tolVar"), opts.tolVar = 1e-4; end

if ~isfield(hp, "alpha"), hp.alpha = 1; end
if ~isfield(hp, "beta"), hp.beta = 1; end
if ~isfield(hp, "mu"), hp.mu = 1; end
if ~isfield(hp, "eta"), hp.eta = 0.99 * hp.mu; end


% data prepare
[idx, col, val] = find(A);
loss = zeros(opts.maxIter, 5);
deltaVar = zeros(opts.maxIter, 3);
VVT = V * V';

% main loop
for iter = 1:opts.maxIter
    % calculating loss
    loss(iter, 1) = 0.5 * norm(S - VVT, "fro") ^ 2;                             % similarity loss
    loss(iter, 2) = hp.beta * sum(D .* VVT, "all");                             % dissimilarity loss
    loss(iter, 3) = hp.eta * sum(w .* p);                                       % discrepancy loss
    loss(iter, 4) = 0.5 * ((hp.mu - 1) * sum(w .^ 2) + hp.mu * sum(p .^ 2));    % density loss

    % Update V
    [T, last_V, norm_V2] = deal(S - hp.beta * D, V, vecnorm(V, 2, 1) .^ 2);
    T = (T + T') / 2;
    for j = 1:size(V, 2)
        V0 = V(:, [1:j-1, j+1:end]);
        V1 = hp.alpha * pinv(V0);

        % orthognoal regularization (approximate value)
        loss(iter, 5) = loss(iter, 5) + (V(:, j)' * V0) * (V1 * V(:, j)) - hp.alpha * norm_V2(j);

        Ev = T * V(:, j) - V0 * (V1 + V0') * V(:, j) + hp.alpha * V(:, j);
        d = max(Ev, 0) / (norm_V2(j) + eps) - V(:, j);
        norm_d2 = norm(d, "fro") ^ 2;

        % if d is too small, indicate that V(:, j) is already convergence
        if norm_d2 <= 1e-8, continue, end

        dEd = d' * T * d - (d' * V0) * ((V1 + V0') * d) + hp.alpha * (d' * d);
        dEv = d' * Ev;
        lambda = getLambda(norm_d2, norm_V2(j), sum(d .* V(:, j)), dEd, dEv);
        V(:, j) = V(:, j) + lambda * d;
    end
    deltaVar(iter, 1) = norm(V - last_V, "fro") / norm(last_V, "fro");
    VVT = V * V';

    % update (w, p)
    [w0, p0, c] = deal(w, p, sum(reshape(VVT(idx) .* val, size(VVT)))');
    for i = 1:20
        w = proj2simplex((c - hp.eta * p) / hp.mu);
        p = proj2simplex(-(hp.beta * c + hp.eta .* w) / hp.mu);
    end
    deltaVar(iter, 2) = norm(w - w0, "fro") / norm(w0, "fro");
    deltaVar(iter, 3) = norm(p - p0, "fro") / norm(p0, "fro");

    % calculate S and D (within O(n^2) complexity)
    S(idx) = val .* w(col);     % S = reshape(full(sum(A .* w, 2)), n, n);
    D(idx) = val .* p(col);     % D = reshape(full(sum(A .* p, 2)), n, n);

    % error handling, return all zeros V as poor solution
    if any(isnan(V), "all") || norm(V, "fro") <= 1e-8
        V(:) = 0;
        break;
    end

    % check terminal condition
    if iter >= 2
        lastLoss = sum(loss(iter - 1, :));
        curLoss = sum(loss(iter, :));
        deltaLoss = abs(lastLoss - curLoss) / abs(lastLoss);

        if sum(deltaVar(iter, :)) <= opts.tolVar || deltaLoss <= opts.tolLoss
            break;
        end
    end
end

% delete redundant rows
loss(iter+1:end, :) = [];
deltaVar(iter+1:end, :) = [];
end

