clc, clear, close all;

[mu, beta, n] = deal(1, 2, 20);
eta = 0.99 * mu;
c = sort(rand(n, 1), "descend");
loss = @(w, p) 0.5 * mu * sum(w .^ 2 + p .^ 2) + sum(c .* (beta * p - w)) + eta * sum(w .* p);

[w, p] = solver(mu, eta, beta, c);
disp("Minimal loss: " + num2str(loss(w, p)));
disp("Check optimal condition: " + num2str(sum(w .* p)) + " (should be zero)");

% running time test
t = timeit(@() solver(mu, eta, beta, c), 2);
disp("Average running time: " + num2str(t) + " second");

% visualization
figure, bar(1:length(w), w, 1, FaceAlpha=0.7); ylabel("$w$", Interpreter="latex");
yyaxis right; bar(1:length(p), p, 1, FaceAlpha=0.7); ylabel("$p$", Interpreter="latex");
xlabel("k-th nearest neighbor"); grid on;


function [w, p] = solver(mu, eta, beta, c)
n = length(c);
[w, p] = deal(rand(n, 1), rand(n, 1));
[w, p] = deal(w / sum(w), p / sum(p));
for i = 1:20
    w = proj2simplex((c - eta * p) / mu);
    p = proj2simplex(-(beta * c + eta .* w) / mu);
end
end