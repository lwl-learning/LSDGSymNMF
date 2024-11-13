function Z = augmentAffinity(S, D, V)
[S, D, Y] = deal((S + S') / 2, (D + D') / 2, V * V');
[S, D, Y] = deal(S / max(S, [], "all"), D / max(D, [], "all"), Y / max(Y, [], "all"));
Z = (Y < D) .* (S .* (1 + Y - D)) + (Y >= D) .* (1 - (1 - S) .* (1 - Y + D));
end