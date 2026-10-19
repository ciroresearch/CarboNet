%% This notebook simulates the closed-loop system with output feedback and uncertainties as 
% explained in [2]. The controller is designed using the nominal matrix A as shown in the MATLAB script 
% "outputFdbk_CO2case.m".

% REFERENCES:
% [1] Ilka, A. and Murgovski, N., 2022. Novel results on output-feedback LQR design. 
% IEEE Transactions on Automatic Control, 68(9), pp. 5187-5200.
% [2] Zocco, F., Haddad, W.M. and Malvezzi, M., 2025. CarboNet: A finite-time combustion-tolerant compartmental network for 
% tropospheric carbon control. arXiv preprint arXiv:2508.16774.
% [3] (last access: 14 November 2025) https://codeocean.com/capsule/4291166/tree/v1 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the nominal matrix A:

A = [-0.2, 0.5, 0.5, 0.1, 0.5; 0.0, -0.7, 0.0, 0.0, 0.0; 0.0, 0.0, -0.4, 0.0, 0.0; 0.2, 0.5, 0.5, -0.1, 0.5; 0.0, 0.0, 0.0, 0.0, -0.2];
B = [-1; 0; 0; 0; 0];
C = [1, 0, 0, 0, 0];
D = 0;

% Create matrix with 10% variation of entries of A:
Delta_10 = 0.1*A;
A_10p = A + Delta_10;
A_10m = A - Delta_10;

% Create matrix with 30% variation of entries of A:
Delta_30 = 0.3*A;
A_30p = A + Delta_30;
A_30m = A - Delta_30;                  

% Set initial conditions x0, set-point xe, and translate the state and input:
x0 = [915.4; 210.0; 500.0; 1830.8; 4]; %initial conditions in tonnes, see [2] for details
x1e = 637.2; %as indicated in the source paper [2], which corresponds to the CO2 mass in the pre-industrial era inside a fixed volume
x2e = 0; %see source paper [2] for the reason of this condition
x3e = 0; %see source paper [2] for the reason of this condition
x4e = (0.2/0.1)*x1e; %see source paper [2] for the reason of this condition 
x7e = 0;
xe = [x1e; x2e; x3e; x4e; x7e]; 
x_tilde_0 = x0 - xe;
v_e = 0;

% Optimal gain computed on the nominal A and with Algorithm 1 (Newton's method):
K = -0.8372;

% Closed-loop matrices with uncertainties:
A_tilde = A - B*K*C; %nominal system
A_tilde_10p = A_10p - B*K*C;
A_tilde_10m = A_10m - B*K*C;
A_tilde_30p = A_30p - B*K*C;
A_tilde_30m = A_30m - B*K*C;

% Closed-loop systems:
ss_closedLoop = ss(A_tilde, zeros(size(B)), C, zeros(size(D)));
ss_closedLoop_10p = ss(A_tilde_10p, zeros(size(B)), C, zeros(size(D)));
ss_closedLoop_10m = ss(A_tilde_10m, zeros(size(B)), C, zeros(size(D)));
ss_closedLoop_30p = ss(A_tilde_30p, zeros(size(B)), C, zeros(size(D)));
ss_closedLoop_30m = ss(A_tilde_30m, zeros(size(B)), C, zeros(size(D)));

%% Plots:
t_final = 60; 
t = linspace(0, t_final, 10^5);
[y_tilde, ~, x_tilde] = initial(ss_closedLoop, x_tilde_0, t);
[y_tilde_10p, ~, x_tilde_10p] = initial(ss_closedLoop_10p, x_tilde_0, t);
[y_tilde_10m, ~, x_tilde_10m] = initial(ss_closedLoop_10m, x_tilde_0, t);
[y_tilde_30p, ~, x_tilde_30p] = initial(ss_closedLoop_30p, x_tilde_0, t);
[y_tilde_30m, ~, x_tilde_30m] = initial(ss_closedLoop_30m, x_tilde_0, t);

% Original state x(t):
% Nominal case:
x1 = x_tilde(:,1) + x1e;
x2 = x_tilde(:,2) + x2e;
x3 = x_tilde(:,3) + x3e;
x4 = x_tilde(:,4) + x4e;
x7 = x_tilde(:,5) + x7e;
% With uncertainty of +10%:
x1_10p = x_tilde_10p(:,1) + x1e;
x2_10p = x_tilde_10p(:,2) + x2e;
x3_10p = x_tilde_10p(:,3) + x3e;
x4_10p = x_tilde_10p(:,4) + x4e;
x7_10p = x_tilde_10p(:,5) + x7e;
%
x1_10m = x_tilde_10m(:,1) + x1e;
x2_10m = x_tilde_10m(:,2) + x2e;
x3_10m = x_tilde_10m(:,3) + x3e;
x4_10m = x_tilde_10m(:,4) + x4e;
x7_10m = x_tilde_10m(:,5) + x7e;
%
x1_30p = x_tilde_30p(:,1) + x1e;
x2_30p = x_tilde_30p(:,2) + x2e;
x3_30p = x_tilde_30p(:,3) + x3e;
x4_30p = x_tilde_30p(:,4) + x4e;
x7_30p = x_tilde_30p(:,5) + x7e;
%
x1_30m = x_tilde_30m(:,1) + x1e;
x2_30m = x_tilde_30m(:,2) + x2e;
x3_30m = x_tilde_30m(:,3) + x3e;
x4_30m = x_tilde_30m(:,4) + x4e;
x7_30m = x_tilde_30m(:,5) + x7e;

figure
plot(t, x1, 'k', 'linewidth', 4);
hold on;
plot(t, x2, 'k', 'linewidth', 4);
hold on;
plot(t, x3, 'k', 'linewidth', 4);
hold on;
plot(t, x4, 'k', 'linewidth', 4);
grid on
plot(t, x7, 'k', 'linewidth', 4);
%
plot(t, x1_10p, 'b', 'linewidth', 4);
hold on;
plot(t, x2_10p, 'b', 'linewidth', 4);
hold on;
plot(t, x3_10p, 'b', 'linewidth', 4);
hold on;
plot(t, x4_10p, 'b', 'linewidth', 4);
grid on
plot(t, x7_10p, 'b', 'linewidth', 4);
%
plot(t, x1_10m, 'r', 'linewidth', 4);
hold on;
plot(t, x2_10m, 'r', 'linewidth', 4);
hold on;
plot(t, x3_10m, 'r', 'linewidth', 4);
hold on;
plot(t, x4_10m, 'r', 'linewidth', 4);
grid on
plot(t, x7_10m, 'r', 'linewidth', 4);
%
plot(t, x1_30p, 'g', 'linewidth', 4);
hold on;
plot(t, x2_30p, 'g', 'linewidth', 4);
hold on;
plot(t, x3_30p, 'g', 'linewidth', 4);
hold on;
plot(t, x4_30p, 'g', 'linewidth', 4);
grid on
plot(t, x7_30p, 'g', 'linewidth', 4);
%
plot(t, x1_30m, 'm', 'linewidth', 4);
hold on;
plot(t, x2_30m, 'm', 'linewidth', 4);
hold on;
plot(t, x3_30m, 'm', 'linewidth', 4);
hold on;
plot(t, x4_30m, 'm', 'linewidth', 4);
grid on
plot(t, x7_30m, 'm', 'linewidth', 4);
grid on
title("Newton's method")
xlabel('Time, t [d]')
ylabel('State, x(t) [t]')
fontsize(35,"points")
legend('Nominal', '', '', '', '', '+10%', '', '', '', '', '-10%', '', '', '', '', '+30%', '', '', '', '', '-30%', '', '', '', '');
xlim([0, 60])
xticks([0, 10, 20, 30, 40, 50, 60, 70])

% Original output y(t):
y = y_tilde + x1e;
y_10p = y_tilde_10p + x1e;
y_10m = y_tilde_10m + x1e;
y_30p = y_tilde_30p + x1e;
y_30m = y_tilde_30m + x1e;
figure
plot(t, y, 'k', 'linewidth', 4)
hold on;
plot(t, y_10p, 'b', 'linewidth', 4)
hold on;
plot(t, y_10m, 'r', 'linewidth', 4)
hold on;
plot(t, y_30p, 'g', 'linewidth', 4)
hold on;
plot(t, y_30m, 'm', 'linewidth', 4)
grid on
title("Newton's method")
xlabel('Time, t [d]')
ylabel('Output, y(t) [t]')
fontsize(35,"points")
legend('Nominal', '+10%', '-10%', '+30%', '-30%');
xlim([0, 60])
xticks([0, 10, 20, 30, 40, 50, 60])
yticks([600, 700, 800, 900, 1000])

% Original control input u(t) = -K*(y - C*xe) + v_e:
n_m = 5;
n_r = 1;

u = - K*(y - C*xe) + v_e;
u_m = (n_m/(n_m+n_r))*u;
u_r = (n_r/(n_m+n_r))*u;
%
u_10p = - K*(y_10p - C*xe) + v_e;
u_m_10p = (n_m/(n_m+n_r))*u_10p;
u_r_10p = (n_r/(n_m+n_r))*u_10p;
%
u_10m = - K*(y_10m - C*xe) + v_e;
u_m_10m = (n_m/(n_m+n_r))*u_10m;
u_r_10m = (n_r/(n_m+n_r))*u_10m;
%
u_30p = - K*(y_30p - C*xe) + v_e;
u_m_30p = (n_m/(n_m+n_r))*u_30p;
u_r_30p = (n_r/(n_m+n_r))*u_30p;
%
u_30m = - K*(y_30m - C*xe) + v_e;
u_m_30m = (n_m/(n_m+n_r))*u_30m;
u_r_30m = (n_r/(n_m+n_r))*u_30m;
figure
plot(t, u_m, 'k', 'linewidth', 4)
hold on
plot(t, u_r, 'k', 'linewidth', 4)
hold on;
plot(t, u_m_10p, 'b', 'linewidth', 4)
hold on
plot(t, u_r_10p, 'b', 'linewidth', 4)
hold on;
plot(t, u_m_10m, 'r', 'linewidth', 4)
hold on
plot(t, u_r_10m, 'r', 'linewidth', 4)
hold on;
plot(t, u_m_30p, 'g', 'linewidth', 4)
hold on
plot(t, u_r_30p, 'g', 'linewidth', 4)
hold on;
plot(t, u_m_30m, 'm', 'linewidth', 4)
hold on
plot(t, u_r_30m, 'm', 'linewidth', 4)
grid on
title("Newton's method")
legend('Nominal', '', '+10%', '', '-10%', '', '+30%', '', '-30%', '');
xlabel('Time, t [d]')
ylabel('u_m(t), u_r(t) [t/d]')
fontsize(35,"points")
xlim([0, 60])
xticks([0, 10, 20, 30, 40, 50, 60])