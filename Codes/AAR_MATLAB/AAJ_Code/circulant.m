clear
close all
restoredefaultpath


n = 200;

A = speye(n-1);
A = [ [zeros(1,n-1) 1]; [A zeros(n-1,1)] ];
A = sparse(A);

b = zeros(n,1);
b(n) = 1;

x = A\b;

tol= 10^(-8);
maxit = 10^6;
restart = size(A,1);
  
M1 = speye(size(A));
M2 = speye(size(A));
M1_inv = inv(M1);
M2_inv = inv(M2);

time_aaj2 = 0.0;
time_gmres2 = 0.0;
   
x_guess = zeros(size(A,1),1);

start_aaj2 = cputime;
[x_aaj2, iter_aaj2, relres_aaj2, sol_aaj2_hist] = AAJ2_full(A,b,x_guess,tol,maxit,M1,M2, 1);
% [x_aaj2, iter_aaj2] = AAJ2(A,b,x_guess,tol,maxit,M1,M2, 0.2, 10, 6);
finish_aaj2 = cputime; 

error_aaj2 = norm(x_aaj2 - x)/norm(x);

time_aaj2 = time_aaj2 + (finish_aaj2 - start_aaj2);

start_gmres2 = cputime;
[ x_gmres2, relres_gmres2, iter_gmres2 ] = gmres_prec ( A, x_guess, b, M1, M2, restart, maxit, tol);
finish_gmres2 = cputime;   

time_gmres2 = time_gmres2 + (finish_gmres2 - start_gmres2);

error_gmres2 = norm(x_gmres2 - x)/norm(x);

display(['Modified Anderson took', ' ', num2str(time_aaj2), ' seconds', ' with error ', num2str(error_aaj2)]);
display(['GMRES2', ' ', num2str(time_gmres2), ' seconds', ' with error ', num2str(error_gmres2)]);

