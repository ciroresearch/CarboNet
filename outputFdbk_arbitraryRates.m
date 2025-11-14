%% Static output-feedback design and simulation of the closed-loop response to initial conditions x_0.
% The output-feedback controller Ki is given by the i-th algorithm
% of Ilka et al. [1], whose MATLAB implementation is available
% in [3]. This script is part of the source code of [2] and covers the case
% of arbitrary rate constants a_{k,s}.

% REFERENCES:
% [1] Ilka, A. and Murgovski, N., 2022. Novel results on output-feedback LQR design. 
% IEEE Transactions on Automatic Control, 68(9), pp. 5187-5200.
% [2] Zocco, F., Haddad, W.M. and Malvezzi, M., 2025. CarboNet: A finite-time combustion-tolerant compartmental network for 
% tropospheric carbon control. arXiv preprint arXiv:2508.16774.
% [3] (last access: 14 November 2025) https://codeocean.com/capsule/4291166/tree/v1 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matrices of LTI system:
A = [-3.40742062, 95.81610494, 39.62507976, 5.6353227; 0, -128.84605851, 0, 0; 0, 0, -95.19704966, 0; 3.40742062, 38.56915493, 63.91448913, -5.6353227];
B = [-1; 0; 0; 0];
C = [1, 0, 0, 0];
D = 0;

% Weighting matrices:
R_1 = C'*C;                          
r_2 = eye(size(B,2));                       
N = zeros(size(A,1),size(B,2));                   

%% Verify 3 of the 4 conditions for stabilizability:
% [Condition 1] The pair (A, B) is stabilizable:
% (check it with the PBH rank test: rank[\sigma*I-A   B] = n  for all eigenvalues of A, namely, \sigma, such that Re(\sigma) >= 0)
% NOTE: the check performed here assumes that A has one eigenvalue equal to 0 and
% three eigenvalues have negative real parts, which is often the case with
% A as defined in [2].
eig_A = eig(A);
PBHtest_stabilizability = [-A, B];
n = size(A,1);
if rank(PBHtest_stabilizability) == n
    disp("The pair (A, B) is stabilizable")
end

%[Condition 2] The pair (A, C) is detectable:
% PBH rank test
PBHtest_detectability = [-A; C];
if rank(PBHtest_detectability) == n
    disp("The pair (A, C) is detectable")
end

%[Condition 3] The weighting matrices must satisfy the equation (6) of Ilka et
%al. [1]:
weightMatrices_1 = [R_1, N];
weightMatrices_2 = [weightMatrices_1; N' r_2];
eig(weightMatrices_2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ss_openLoop = ss(A,B,C,D); 
[~,Pi,~]=lqr(ss_openLoop, R_1, r_2, N);   % initial Lyapunov matrix, calculated 
                                   % as the state-feedback solution
maxIteration = 9e6;                % maximal iteration number
stopCrit = 1e-12;                  % stopping criteria

% Set initial conditions x0, set-point xe, and translate the state and input:
x0 = [1; 2; 3; 4]; %initial conditions
x1e = randi(5); 
x2e = 0; %see source paper [2] for the reason of this condition
x3e = 0; %see source paper [2] for the reason of this condition
x4e = (3.40742062/5.6353227)*x1e; %(a_41/a_14)*x1e; see source paper [2] for the reason of this condition  
xe = [x1e; x2e; x3e; x4e]; 
x_tilde_0 = x0 - xe;
v_e = 0;



%% Output stabilization with Algorithm 1 of Ilka et al. [1]:
% Compute feedback gain:  
startTimeAlg1 = tic;
[K1,P1,iteration1,critFun1]=algorithm1(A,B,C,R_1,r_2,N,Pi,maxIteration,stopCrit);
stoptTimeAlg1 = toc(startTimeAlg1);

% Verify the 4th and last condition for stabilizability:
G1 = K1*C - r_2^(-1)*(B'*P1 + N');
% [Condition 4]:
it_must_be_zero1 = A'*P1 + P1*A + R_1 + G1'*r_2*G1 - (P1*B+N)*r_2^(-1)*(B'*P1+N');
disp(it_must_be_zero1)

% Closed-loop system:
A_hat1 = A - B*K1*C;
ss_closedLoop1 = ss(A_hat1, zeros(size(B)), C, zeros(size(D)));

% Plots:
% Translated states (i.e., with origin in x_e):
t_final = 70; 
t = linspace(0, t_final, 10^5);
[y_tilde1, ~, x_tilde1] = initial(ss_closedLoop1, x_tilde_0, t);
figure
plot(t, x_tilde1)
grid on
title("Newton's method")
leg = legend('$\tilde{x}_1$', '$\tilde{x}_2$', '$\tilde{x}_3$', '$\tilde{x}_4$');
set(leg,'Interpreter','latex');

% Translated output y_tilde(t) = C*x_tilde(t):
figure
plot(t, y_tilde1)
grid on
title("Newton's method")
leg = legend('$\tilde{y} = \tilde{x}_1$');
set(leg,'Interpreter','latex');

% Original state x(t):
x11 = x_tilde1(:,1) + x1e;
x21 = x_tilde1(:,2) + x2e;
x31 = x_tilde1(:,3) + x3e;
x41 = x_tilde1(:,4) + x4e;
figure
plot(t, x11, 'linewidth',4);
hold on;
plot(t, x21, '--', 'linewidth',4);
hold on;
plot(t, x31, '--', 'linewidth',4);
hold on;
plot(t, x41, 'linewidth',4);
grid on
title("Newton's method")
xlabel('Time, t [d]')
ylabel('Original state, x(t) [t]')
fontsize(29,"points")
leg = legend('$x_1$', '$x_2$', '$x_3$', '$x_4$');
set(leg,'Interpreter','latex');
xlim([0, 15])
xticks([0, 5, 10, 15])

% Original output y(t):
y1 = y_tilde1 + x1e;
figure
plot(t, y1, 'linewidth', 4)
grid on
title("Newton's method")
xlabel('Time, t [d]')
ylabel('Original output, y(t) [t]')
fontsize(29,"points")
xlim([0, 15])
xticks([0, 5, 10, 15])

% Original control input u(t) = -K*(y - C*xe) + v_e:
u1 = - K1*(y1 - C*xe) + v_e;
figure
plot(t, u1, 'linewidth', 4)
grid on
title("Newton's method")
xlabel('Time, t [d]')
ylabel('Control input, u(t) [t/d]')
fontsize(29,"points")
xlim([0, 15])
xticks([0, 5, 10, 15])



%% Output stabilization with Algorithm 2 of Ilka et al. [1]:
% Compute feedback gain:  
startTimeAlg2 = tic;
[K2,P2,iteration2,critFun2]=algorithm2(A,B,C,R_1,r_2,N,Pi,maxIteration,stopCrit);
stoptTimeAlg2 = toc(startTimeAlg2);

% Verify the 4th and last condition for stabilizability:
G2 = K2*C - r_2^(-1)*(B'*P2 + N');
% [Condition 4]:
it_must_be_zero2 = A'*P2 + P2*A + R_1 + G2'*r_2*G2 - (P2*B+N)*r_2^(-1)*(B'*P2+N');
disp(it_must_be_zero2)

% Closed-loop system:
A_hat2 = A - B*K2*C;
ss_closedLoop2 = ss(A_hat2, zeros(size(B)), C, zeros(size(D)));

% % Plots:
% % Translated states (i.e., with origin in x_e):
% [y_tilde2, ~, x_tilde2] = initial(ss_closedLoop2, x_tilde_0);
% figure
% plot(x_tilde2)
% grid on
% title("Modif. Newton's method")
% leg = legend('$\tilde{x}_1$', '$\tilde{x}_2$', '$\tilde{x}_3$', '$\tilde{x}_4$');
% set(leg,'Interpreter','latex');
% 
% % Translated output y_tilde(t) = C*x_tilde(t):
% figure
% plot(y_tilde2)
% grid on
% title("Modif. Newton's method")
% leg = legend('$\tilde{y} = \tilde{x}_1$');
% set(leg,'Interpreter','latex');
% 
% % Original state x(t):
% x12 = x_tilde2(:,1) + x1e;
% x22 = x_tilde2(:,2) + x2e;
% x32 = x_tilde2(:,3) + x3e;
% x42 = x_tilde2(:,4) + x4e;
% figure
% plot(x12, 'linewidth',4);
% hold on;
% plot(x22, '--', 'linewidth',4);
% hold on;
% plot(x32, '--', 'linewidth',4);
% hold on;
% plot(x42, 'linewidth',4);
% grid on
% title("Modif. Newton's method")
% xlabel('Time, t')
% ylabel('Original state, x(t)')
% fontsize(24,"points")
% leg = legend('$x_1$', '$x_2$', '$x_3$', '$x_4$');
% set(leg,'Interpreter','latex');
% 
