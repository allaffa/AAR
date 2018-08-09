clear
close all
restoredefaultpath

problem = 'nos3';

directory = strcat('./tests/', problem);
preconditioner = 'diagonal';

addpath(directory);

load('A.mat')

x = rand(size(A,1),1);

b = A * x;

tol= 10^(-8);
maxit = 10^2;
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

elseif  strcmp(preconditioner, 'ainv')   
    
    load('prec_z')
    Z_t=spconvert(prec_z);
    Z_t=sparse(Z_t);
    load('prec_w')
    W=spconvert(prec_w);
    W=sparse(W);
    nn=size(Z_t,1);
    W=W(1:nn,1:nn);
    Z_t=Z_t(1:nn,1:nn);
    A_sp1=A(1:nn,1:nn);

    D_inv=sparse(diag(diag(Z_t)));
    Z=Z_t'*inv(D_inv);
    C=sparse(Z * D_inv * W);

    M1_inv = W;
    M2_inv = Z * D_inv;
    
    M1 = speye(size(A));
    M2 = speye(size(A));
    
 elseif  strcmp(preconditioner, 'ilu')    
    
    setup.type='nofill';
    setup.milu='off';

    [M1,M2]=ilu(A,setup);
    
end

% time_aaj = 0.0;
time_aaj2 = 0.0;
   
x_guess = rand(size(A,1),1);

start_aaj2 = cputime;
% [x_aaj2, iter_aaj2, relres_aaj2, sol_aaj2_hist] = AAJ2(A,b,x_guess,tol,maxit,M1,M2, 1, 10, 6);
% [x_aaj2, iter_aaj2, relres_aaj2, sol_aaj2_hist] = AAJ2_modified(A,b,x_guess,tol,maxit,M1,M2, 0.2, 10, 6);
% [x_aaj2, iter_aaj2, relres_aaj2, sol_aaj2_hist] = AAJ2_random(A,b,x_guess,tol,maxit,M1,M2, 0.2, 10, 6);
[x_aaj2, iter_aaj2, relres_aaj2, sol_aaj2_hist, spectrum2] = AAJ2_spectrum(A,b,x_guess,tol,maxit,M1,M2, 1, 10, 6);
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
title(['Relative residual for ', '', problem]);
set(gca, 'fontsize', 18)

figure()
semilogy(error_hist_aaj2, '-o', 'linewidth', 2);
xlabel('Number of iterations');
ylabel('Relative error');
title(['Relative error for ', '', problem]);
set(gca, 'fontsize', 18)

ismonotonic(relres_aaj2, 0, 'DECREASING')

save(strcat('./results/', problem, '.mat'));
