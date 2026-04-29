clc; clear; close all;

%% Parameters
k = 27000;           % N/m
beta_c = 0.002;      % m
beta_e = 0.002;      % m
F_fric = 20;         % N

%% Generate motion
t = linspace(0, 2*pi, 1000);
delta = 0.02 * sin(t);          % +/- 20 mm
delta_dot = 0.02 * cos(t);

%% Initialize force
F = zeros(size(delta));

%% Compute force with hysteresis
for i = 1:length(delta)
    if delta_dot(i) >= 0
        % Extension (rebound)
        F(i) = k * (delta(i) + beta_e) + F_fric;
    else
        % Compression (jounce)
        F(i) = k * (delta(i) - beta_c) - F_fric;
    end
end

%% Linear reference
F_linear = k * delta;

%% Plot Force vs Displacement
figure;
plot(delta*1000, F_linear, 'k--', 'LineWidth', 1.5); hold on;
plot(delta*1000, F, 'r', 'LineWidth', 2);

grid on;
xlabel('Displacement (mm)');
ylabel('Force (N)');
title('Spring Hysteresis with Friction Offset');
legend('Linear Spring', 'With Hysteresis');

%% Plot Force vs Time
figure;
plot(t, F, 'b', 'LineWidth', 1.5); hold on;
plot(t, F_linear, 'k--');

grid on;
xlabel('Time');
ylabel('Force (N)');
title('Force vs Time');
legend('Hysteresis Spring', 'Linear Spring');
