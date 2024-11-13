function [V, loss, deltaVar] = PHALS(S, V, opts)
% Default algorithm options
if ~isfield(opts, "maxIter"), opts.maxIter = 1000; end
if ~isfield(opts, "tolLoss"), opts.tolLoss = 1e-4; end
if ~isfield(opts, "tolVar"), opts.tolVar = 1e-4; end

% data prepare
[loss, deltaVar] = deal(zeros(opts.maxIter, 1));
S = (S + S') / 2;               % ensure the symmetric
V = sqrt(sum(S .* (V * V'), "all")) * V / norm(V * V', "fro");    % scaling V
loss(1) = 0.5 * norm(S - V * V', "fro") ^ 2;

% main loop
for iter = 1:opts.maxIter
    [last_V, norm_V2, SV] = deal(V, vecnorm(V, 2, 1) .^ 2, S * V);

    % update V(:, j)
    for j = 1:size(V, 2)
        V0 = V(:, [1:j-1, j+1:end]);
        Ev = SV(:, j) - V0 * (V0' * V(:, j));
        d = max(Ev, 0) / norm_V2(j) - V(:, j);
        norm_d2 = norm(d, "fro") ^ 2;

        % if d is too small, indicate that V(:, j) is already convergence
        if norm_d2 <= 1e-8 || any(isnan(d)), continue, end    

        dEd = d' * S * d - (d' * V0) * (V0' * d);
        dEv = d' * Ev;
        lambda = getLambda(norm_d2, norm_V2(j), sum(d .* V(:, j)), dEd, dEv);
        V(:, j) = V(:, j) + lambda * d;   % Update Vj
    end

    % check terminal condition
    loss(iter + 1) = 0.5 * norm(S - V * V', "fro") ^ 2;
    deltaVar(iter) = norm(V - last_V, "fro") / norm(last_V, "fro");
    
    if deltaVar(iter) <= opts.tolVar || (loss(iter) - loss(iter + 1)) / loss(iter) <= opts.tolLoss
        break;
    end
end

loss(iter+1:end) = [];          % delete redundant rows
deltaVar(iter+1:end) = [];      % delete redundant rows
end

