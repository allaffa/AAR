clear
close all
restoredefaultpath

preconditioner = 'diagonal';

N = 100;

A = 2 * diag(ones(N,1)) -1 * diag(ones(N-1,1),1) -1 * diag(ones(N-1,1),-1);
A = [0 zeros(1,N) 0; zeros(N,1) A zeros(N,1); zeros(1,N+2)];
A = sparse(A);
A(1,1) = 1;
A(N+2,N+2) = 1;

b = [ 0; ones(size(A,1)-2,1); 0] * (1/(size(A,1)))^2; % homogeneous Dirichlet
x = A\b;

tol= 10^(-8);
maxit = 10^5;
restart = size(A,1);

if strcmp(preconditioner, 'none')
    
    M1 = speye(size(A));
    M2 = speye(size(A));
    M1_inv = inv(M1);
    M2_inv = inv(M2);

elseif strcmp(preconditioner, 'diagonal')
    
    M1 = speye(size(A));
    M2 = diag(diag(A));
    M1_inv = inv(M1);
    M2_inv = inv(M2);
    
elseif strcmp(preconditioner, 'block')
    
    size_block = 5;
    n_blocks = size(A,1)/size_block;

    D_inv = sparse(size(A));
    D = sparse(size(A));

    for i = 0:n_blocks-1
        D(i*size_block+1:(i+1)*size_block,i*size_block+1:(i+1)*size_block) = A(i*size_block+1:(i+1)*size_block,i*size_block+1:(i+1)*size_block);
        D_inv(i*size_block+1:(i+1)*size_block,i*size_block+1:(i+1)*size_block) = inv(A(i*size_block+1:(i+1)*size_block,i*size_block+1:(i+1)*size_block));
    end

    D = sparse(D);
    D_inv = sparse(D_inv);
    
    M1 = speye(size(A));
    M1_inv = inv(M1);
    M2 = D;
    M2_inv = D_inv;
    
end

% time_aaj = 0.0;
time_aaj2 = 0.0;
   
x_guess = rand(size(A,1),1);

start_aaj2 = cputime;
[x_aaj2, iter_aaj2, relres_aaj2, sol_aaj2_hist] = AAJ2(A,b,x_guess,tol,maxit,M1,M2, 1, 1, 10, 6);
finish_aaj2 = cputime;    

time_aaj2       = time_aaj2 + (finish_aaj2 - start_aaj2);

error_aaj2 = norm(x - x_aaj2)/norm(x);

error_hist_aaj2 = [];

for iter = 1: size(sol_aaj2_hist, 2)
    
  error_hist_aaj2 = [error_hist_aaj2 norm(x - sol_aaj2_hist(:,iter))/norm(x)];  
    
end

% display(['Anderson took', ' ', num2str(time_aaj), ' seconds', ' with error ', num2str(error_aaj)]);
display(['Modified Anderson took', ' ', num2str(time_aaj2), ' seconds', ' with error ', num2str(error_aaj2)]);

figure()
semilogy(relres_aaj2, '-o', 'linewidth', 2);
xlabel('Number of iterations');
ylabel('Relative residual');
title(['Relative residual']);

figure()
semilogy(error_hist_aaj2, '-o', 'linewidth', 2);
xlabel('Number of iterations');
ylabel('Relative error');
title(['Relative error']);

ismonotonic(relres_aaj2, 0, 'DECREASING')
