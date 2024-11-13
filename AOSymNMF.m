function [V, loss, deltaVar] = AOSymNMF(S, V, opts, hp)
% Default algorithm options
if nargin < 3, opts = struct(); end
if ~isfield(opts, "maxIter"), opts.maxIter = 1000; end
if ~isfield(opts, "tolLoss"), opts.tolLoss = 1e-4; end
if ~isfield(opts, "tolVar"), opts.tolVar = 1e-4; end

if nargin < 4, hp = struct(); end
if ~isfield(hp, "alpha"), hp.alpha = 1; end

% data prepare
loss = zeros(opts.maxIter, 2);
deltaVar = zeros(opts.maxIter, 1);
S = (S + S') / 2;               % ensure the symmetric
V = sqrt(sum(S .* (V * V'), "all")) * V / norm(V * V', "fro");    % scaling V

% main loop
for iter = 1:opts.maxIter
    loss(iter, 1) = 0.5 * norm(S - V * V', "fro") ^ 2;

    [last_V, norm_V2, SV] = deal(V, vecnorm(V, 2, 1) .^ 2, S * V);

    % update V(:, j)
    for j = 1:size(V, 2)
        V0 = V(:, [1:j-1, j+1:end]);
        Vp = hp.alpha * pinv(V0);
        Ev = SV(:, j) - V0 * ((Vp + V0') * V(:, j)) + hp.alpha * V(:, j);
        loss(iter, 2) = loss(iter, 2) - hp.alpha * norm_V2(j) + (V(:, j)' * V0) * (Vp * V(:, j));
        d = max(Ev, 0) / norm_V2(j) - V(:, j);
        norm_d2 = norm(d, "fro") ^ 2;

        % if d is too small, indicate that V(:, j) is already convergence
        if norm_d2 <= 1e-8 || any(isnan(d)), continue, end    

        dEd = d' * S * d - (d' * V0) * ((Vp + V0') * d) + hp.alpha * norm_d2;
        dEv = d' * Ev;
        lambda = getLambda(norm_d2, norm_V2(j), sum(d .* V(:, j)), dEd, dEv);
        V(:, j) = V(:, j) + lambda * d;   % Update Vj
    end

    % check terminal condition
    deltaVar(iter) = norm(V - last_V, "fro") / norm(last_V, "fro");
    
    if iter >= 2
        lastLoss = sum(loss(iter - 1, :));
        curLoss = sum(loss(iter, :));
        deltaLoss = abs(lastLoss - curLoss) / abs(lastLoss);

        if deltaVar(iter) <= opts.tolVar || deltaLoss <= opts.tolLoss
            break;
        end
    end
end

% delete redundant rows
loss(iter+1:end, :) = []; 
deltaVar(iter+1:end) = [];
end