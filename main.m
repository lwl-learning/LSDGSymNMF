clc, clear, close all;

% --- Global settings
addpath("ClusteringMeasure\");
RS = RandStream("twister", Seed=666);  % for reproducibility
opts = struct("maxIter", 1000, "tolLoss", 1e-4, "tolVar", 1e-4);
hp = struct("alpha", 0.1, "beta", 1, "mu", 0.5);
numSpectralTest = 20;

set(groot, "defaultAxesXGrid", "on");   % active x-axes grid
set(groot, "defaultAxesYGrid", "on");   % active y-axes grid

% --- Load dataset
load("ORL_32x32.mat", "fea", "gnd");
[X, gnd] = deal(reshape(fea', 32 * 32, 400) / 255, gnd');
[n, r] = deal(400, 40);
k = floor(log2(n)) + 1;
[~, ~, ~, A] = get_all_affinities(X);

% --- Initialization
[w, p] = deal(zeros(n, 1));
[w(1:k), p(k+1:end)] = deal(1 / k, 1 / (n - k));
hp.eta = 0.99 * hp.mu;

S0 = reshape(full(sum(A .* w', 2)), [n, n]);
D0 = reshape(full(sum(A .* p', 2)), [n, n]);
V0 = rand(RS, n, r);
V = PHALS(S0, V0, opts);

%% Proposed method
% --- Model training
[V, w, p, S, D, loss, deltaVar] = LSDGSymNMF(A, V, w, p, S0, D0, opts, hp);

% --- Clustering metrics
Z = augmentAffinity(S, D, V);
[ACC, NMI] = deal(zeros(numSpectralTest, 1));
for i = 1:numSpectralTest
    C = SpectralClustering(Z, r);   % clustering result
    ACC(i) = sum(gnd == bestMap(gnd, C)') / n;
    [~, NMI(i)] = compute_nmi(gnd, C);
end
disp("Proposed method:");
disp("ACC mean: " + num2str(mean(ACC)) + ", std: " + num2str(std(ACC)));
disp("NMI mean: " + num2str(mean(NMI)) + ", std: " + num2str(std(NMI)));

% --- Visualization for S, D, V, Z
figure, tiledlayout(1, 4, "TileSpacing", "compact", "Padding", "compact");
nexttile, imagesc(S), xticks([]), yticks([]), title("$S$", "Interpreter", "latex"), colorbar();
nexttile, imagesc(D), xticks([]), yticks([]), title("$D$", "Interpreter", "latex"), colorbar();
nexttile, imagesc(V * V'), xticks([]), yticks([]), title("$V V^T$", "Interpreter", "latex"), colorbar();
nexttile, imagesc(Z), xticks([]), yticks([]), title("$Z$", "Interpreter", "latex"), colorbar();

% --- Visualization for w, p and ACC
onehot = onehotencode(gnd', 2, "ClassNames", 1:r);
gndPC = onehot * onehot';
A(A > 0) = 1;
CR = arrayfun(@(k) full(sum(A(:, k) .* gndPC(:), "all")) / n, 1:n);
[w, p] = deal(w / max(w), p / max(p));

figure, yyaxis left;
bar(1:n, w, 1, "FaceAlpha", 0.7, "FaceColor", "#0072BD"), hold on;
bar(1:n, -p, 1, "FaceAlpha", 0.7, "FaceColor", "#D95319");
yticks(-1:0.5:1); yticklabels([1, 0.5, 0, 0.5, 1]);

yyaxis right; plot(1:n, 2 * (CR - 0.5), LineWidth=1.5, Color="#7E2F8E");
xscale("log"); xlim([1, n]);
yticks(-1:0.5:1); yticklabels(0:0.25:1);
legend(["$w$", "$p$", "Neighbors' ACC"], "interpreter", "latex");

% --- Visualization for loss
figure, tiledlayout(2, 3, "TileSpacing", "compact", "Padding", "compact");
nexttile, plot(sum(loss, 2)), title("Total");
nexttile, plot(loss(:, 1)), title("$\frac{1}{2}\|S(w) - V V^T\|_F^2$", "Interpreter", "latex");
nexttile, plot(loss(:, 2)), title("$\beta \langle D(p), V V^T \rangle$", "Interpreter", "latex");
nexttile, plot(loss(:, 3)), title("$\eta \langle S(w), D(p) \rangle$", "Interpreter", "latex");
nexttile, plot(loss(:, 4)), title("$\frac{\mu - 1}{2} \|S(w) \|_F^2 + \frac{\mu}{2} \|D(p) \|_F^2$", "Interpreter", "latex");
nexttile, plot(loss(:, 5)), title("$-\alpha R(V)$", "Interpreter", "latex");

% --- Visualization for deltaVars
figure, tiledlayout(1, 3, "TileSpacing", "compact", "Padding", "compact");
nexttile, plot(deltaVar(:, 1)), title("$V$", "Interpreter", "latex");
nexttile, plot(deltaVar(:, 2)), title("$w$", "Interpreter", "latex");
nexttile, plot(deltaVar(:, 3)), title("$p$", "Interpreter", "latex");


%% AOSymNMF
% --- Model training
[V, loss, deltaVar] = AOSymNMF(S0, V0, opts, hp);   % Note: we use same initialized V0 and S0

% --- Clustering metrics
[~, C] = max(V, [], 2);
ACC = sum(gnd == bestMap(gnd, C)') / n;
[~, NMI] = compute_nmi(gnd, C);
disp("AOSymNMF:");
disp("ACC: " + num2str(ACC));
disp("NMI: " + num2str(NMI));

% --- Visualization for loss
figure, tiledlayout(1, 3, "TileSpacing", "compact", "Padding", "compact");
nexttile, plot(sum(loss, 2)), title("Total");
nexttile, plot(loss(:, 1)), title("$\frac{1}{2}\|S - V V^T\|_F^2$", "Interpreter", "latex");
nexttile, plot(loss(:, 2)), title("$-\alpha R(V)$", "Interpreter", "latex");

% --- Visualization for deltaVars
figure, plot(deltaVar), title("$V$", "Interpreter", "latex");