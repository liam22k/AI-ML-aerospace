clc; close all; clear;

x = 1;  y = 1;
p0 = [0.5, 0.1, 2, 0.1];            % [w1, b1, w2, b2]
L  = @(p) loss_and_grad(p, x, y);   % loss only (first output)

%% Gradient check
[L0, g_bp] = loss_and_grad(p0, x, y);
h = 1e-6;
g_num = zeros(1, length(p0));
for k = 1:length(p0)
    pPlus  = p0;  pPlus(k)  = pPlus(k)  + h;
    pMinus = p0;  pMinus(k) = pMinus(k) - h;
    g_num(k) = (L(pPlus) - L(pMinus)) / (2*h);
end
rel_err = abs(g_bp - g_num) ./ max(abs(g_bp), abs(g_num));
names = ["w1","b1","w2","b2"];
for k = 1:length(p0)
    fprintf("dL/d%s: backprop = %+.5f  numeric = %+.5f  rel err = %.1e\n", ...
            names(k), g_bp(k), g_num(k), rel_err(k));
end

%% One-step learning-rate sweep
for eta = [0.1 0.2 0.3 0.4 0.5]
    p_new = p0 - eta * g_bp;
    fprintf("eta = %.1f: L = %.5f\n", eta, L(p_new));
end

%% Training loop
eta = 0.1;
p = p0;
for i = 1:5
    [L_old, grad] = loss_and_grad(p, x, y);   % gradient at CURRENT p
    p = p - eta * grad;                       % step and overwrite
    fprintf("step %d: old loss = %.3e, new loss = %.3e\n", i, L_old, L(p));
end

%% Local functions (must be at end of script)
function [L_val, grad] = loss_and_grad(p, x, y)
    % forward pass
    z1    = p(1)*x + p(2);
    a1    = tanh(z1);
    yHat  = p(3)*a1 + p(4);             % linear output
    L_val = 0.5*(yHat - y)^2;
    % backward pass
    delta2 = yHat - y;                  % dL/dyHat
    delta1 = p(3) * delta2 * (1 - a1.^2);
    grad   = [delta1*x, delta1, delta2*a1, delta2];
end
