clear
close all
restoredefaultpath

A1 = [1 1 1; 0 1 3; 0 0 1];

b1 = [2 -4 1]'; % homogeneous Dirichlet
x = A1\b1;

M1 = speye(3);
M2 = speye(3);

tol= 10^(-8);
maxit = 10^3;
restart = 2;

% time_aaj = 0.0;
time_aaj2 = 0.0;
time_gmres2 = 0.0;
   
x_guess = zeros(size(A1,1),1);

start_aaj2 = cputime;
% [x_aaj2, iter_aaj2, relres_aaj2, sol_aaj2_hist] = AAJ2(A1,b1,x_guess,tol,maxit,M1,M2,2/norm(A1, 'inf'),1,2,2);
[x_aaj2, iter_aaj2, relres_aaj2, sol_aaj2_hist] = AAJ2_restart(A1,b1,x_guess,tol,maxit,M1,M2,1,restart+1,1,restart);
% [x_aaj2, iter_aaj2, relres_aaj2, sol_aaj2_hist] = AAJ2_correct(A1,b1,x_guess,tol,maxit,M1,M2,1,1,restart+4,restart);
finish_aaj2 = cputime;    

time_aaj2       = time_aaj2 + (finish_aaj2 - start_aaj2);

error_aaj2 = norm(x - x_aaj2)/norm(x);

error_hist_aaj2 = [];

for iter = 1: size(sol_aaj2_hist, 2)
    
  error_hist_aaj2 = [error_hist_aaj2 norm(x - sol_aaj2_hist(:,iter))/norm(x)];  
    
end

start_gmres2 = cputime;
[ x_gmres2, relres_gmres2, iter_gmres2, flag, sol_hist_gmres2 ] = gmres_prec ( A1, x_guess, b1, M1, M2, restart, maxit, tol);
finish_gmres2 = cputime;

time_gmres2       = time_gmres2 + (finish_gmres2 - start_gmres2);

error_gmres2 = norm(x - x_gmres2)/norm(x);

error_hist_gmres2 = [];

for iter = 1: size(sol_hist_gmres2, 2)
    
  error_hist_gmres2 = [error_hist_gmres2 norm(x - sol_hist_gmres2(:,iter))/norm(x)];  
    
end

% display(['Anderson took', ' ', num2str(time_aaj), ' seconds', ' with error ', num2str(error_aaj)]);
display(['Modified Anderson took', ' ', num2str(time_aaj2), ' seconds', ' with error ', num2str(error_aaj2)]);
display(['GMRES2 took', ' ', num2str(time_gmres2), ' seconds', ' with error ', num2str(error_gmres2)]);


figure()
semilogy(relres_aaj2, '-o', 'linewidth', 2);
xlabel('Number of iterations');
ylabel('Relative residual');
title(['AAR: Relative residual']);

figure()
semilogy(error_hist_gmres2, '-o', 'linewidth', 2);
xlabel('Number of iterations');
ylabel('Relative residual');
title(['GMRES: Relative error']);

figure()
semilogy(error_hist_aaj2, '-o', 'linewidth', 2);
xlabel('Number of iterations');
ylabel('Relative error');
title(['AAR: Relative error']);

ismonotonic(relres_aaj2, 0, 'DECREASING')

%% 

A2 = [1 2 -2; 0 2 4; 0 0 3];

b2 = [3 1 1]'; % homogeneous Dirichlet
x = A2\b2;

M1 = speye(3);
M2 = speye(3);

tol= 10^(-8);
maxit = 10^3;
restart = 2;

% time_aaj = 0.0;
time_aaj2 = 0.0;
time_gmres2 = 0.0;
   
x_guess = zeros(size(A2,1),1);

start_aaj2 = cputime;
% [x_aaj2, iter_aaj2, relres_aaj2, sol_aaj2_hist] = AAJ2(A2,b2,x_guess,tol,maxit,M1,M2,2/norm(A2, 'inf'),1,6,2);
[x_aaj2, iter_aaj2, relres_aaj2, sol_aaj2_hist] = AAJ2_restart(A2,b2,x_guess,tol,maxit,M1,M2,1,restart+1,1,restart);
finish_aaj2 = cputime;    

time_aaj2       = time_aaj2 + (finish_aaj2 - start_aaj2);

error_aaj2 = norm(x - x_aaj2)/norm(x);

error_hist_aaj2 = [];

for iter = 1: size(sol_aaj2_hist, 2)
    
  error_hist_aaj2 = [error_hist_aaj2 norm(x - sol_aaj2_hist(:,iter))/norm(x)];  
    
end

start_gmres2 = cputime;
[ x_gmres2, relres_gmres2, iter_gmres2, flag, sol_hist_gmres2 ] = gmres_prec ( A2, x_guess, b2, M1, M2, restart, maxit, tol);
finish_gmres2 = cputime;

time_gmres2       = time_gmres2 + (finish_gmres2 - start_gmres2);

error_gmres2 = norm(x - x_gmres2)/norm(x);

error_hist_gmres2 = [];

for iter = 1: size(sol_hist_gmres2, 2)
    
  error_hist_gmres2 = [error_hist_gmres2 norm(x - sol_hist_gmres2(:,iter))/norm(x)];  
    
end

% display(['Anderson took', ' ', num2str(time_aaj), ' seconds', ' with error ', num2str(error_aaj)]);
display(['Modified Anderson took', ' ', num2str(time_aaj2), ' seconds', ' with error ', num2str(error_aaj2)]);
display(['GMRES2 took', ' ', num2str(time_gmres2), ' seconds', ' with error ', num2str(error_gmres2)]);


figure()
semilogy(relres_aaj2, '-o', 'linewidth', 2);
xlabel('Number of iterations');
ylabel('Relative residual');
title(['AAR: Relative residual']);

figure()
semilogy(error_hist_gmres2, '-o', 'linewidth', 2);
xlabel('Number of iterations');
ylabel('Relative residual');
title(['GMRES: Relative error']);

figure()
semilogy(error_hist_aaj2, '-o', 'linewidth', 2);
xlabel('Number of iterations');
ylabel('Relative error');
title(['AAR: Relative error']);

ismonotonic(relres_aaj2, 0, 'DECREASING')

