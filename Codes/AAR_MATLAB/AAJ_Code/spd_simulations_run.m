clear 
close all
restoredefaultpath

maxNumCompThreads(1)

problem = 'qa8fm';

directory = strcat('./tests/', problem);
preconditioner = 'ic';

addpath(directory);

load('A.mat')

tol= 10^(-8);
omega = 0.2;
beta = 1;
maxit = 10^5;
restart = 10;
reorder = 0;
r = [];
Perm_ilutp = speye(size(A));
Perm_rcm = speye(size(A));

number_runs = 10;

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
    
    size_block = 3;
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
    
 elseif  strcmp(preconditioner, 'ic')    
   
    if (reorder ==1)
        r = symrcm(A);
        Perm_rcm = speye(size(A));
        Perm_rcm = Perm_rcm(r,r);
        A = A(r,r);
    end     
     
%    [M1]=ichol(A);
    [M1]=ichol(A, struct('type','ict','droptol',1e-4,'diagcomp',0));
    M2 = M1';
    
end

fprintf('CONSTRUCTION OF THE PRECONDITIONER FINISHED \n\n');

time_richardson = 0.0;
time_aaj = 0.0;
time_aaj2 = 0.0;
time_pcg = 0.0;

error_richardson = 0.0;
error_aaj = 0.0;
error_aaj2 = 0.0;
error_pcg = 0.0;

iter_richardson_average = 0.0;
iter_aaj_average = 0.0;
iter_aaj2_average = 0.0;
iter_pcg_average = 0.0; 


for runs = 1:number_runs
    
    x = rand(size(A,1),1);

    b = A * x;
    
    x_guess = rand(size(A,1),1);
    
%     start_richardson = cputime;
%     [x_rich, iter_rich, relres_rich] = Richardson( A, b, 1, x_guess, tol, maxit, M1, M2 );
%     finish_richardson = cputime;

    start_aaj2 = cputime;
    [x_aaj2, iter_aaj2, relres_aaj2] = AAJ2(A,b,x_guess,tol,maxit,M1,M2,omega,beta,10,1);
%     [x_aaj2, iter_aaj2, relres_aaj2] = AAJ2_full(A,b,x_guess,tol,maxit,M1,M2,1);
%     [x_aaj2, iter_aaj2, relres_aaj2] = AAJ2_restart(A,b,x_guess,tol,maxit,M1,M2, 1, maxit, 1, 10);
%     [x_aaj2, iter_aaj2, relres_aaj2] = AAJ2_truncated(A,b,x_guess,tol,maxit,M1,M2,1,10);
%     [x_aaj2, iter_aaj2, relres_aaj2] = AAJ2_error_minimization(A,b,x_guess,tol,maxit,M1,M2,2/norm(A, 'inf'),1,100,6);
    finish_aaj2 = cputime;    


    start_pcg = cputime;
    [x_pcg,~,~,iter_pcg,relres_pcg] = pcg(A,b,tol,maxit,M1,M2,x_guess);
    finish_pcg = cputime;   


%     time_richardson = time_richardson + (finish_richardson - start_richardson);
%     time_aaj        = time_aaj + (finish_aaj - start_aaj);
    time_aaj2       = time_aaj2 + (finish_aaj2 - start_aaj2);
    time_pcg     = time_pcg + (finish_pcg - start_pcg);
%     
%     error_richardson = error_richardson + norm(x_rich - x)/norm(x);
%     error_aaj        = error_aaj + norm(x_aaj - x)/norm(x);
    error_aaj2       = error_aaj2 + norm(x_aaj2 - x)/norm(x);
    error_pcg      = error_pcg + norm(x_pcg - x)/norm(x);
%     
%     iter_richardson_average = iter_richardson_average + iter_rich;
%     iter_aaj_average = iter_aaj_average + iter_aaj;
    iter_aaj2_average = iter_aaj2_average + iter_aaj2;
    iter_pcg_average = iter_pcg_average + iter_pcg;

end


% time_richardson = time_richardson / number_runs;
% time_aaj        = time_aaj / number_runs;
time_aaj2       = time_aaj2 / number_runs;
time_pcg     = time_pcg / number_runs;
% 
% error_richardson = error_richardson / number_runs;
% error_aaj        = error_aaj / number_runs;
error_aaj2       = error_aaj2 / number_runs;
error_pcg     = error_pcg / number_runs;
% 
% iter_richardson_average = iter_richardson_average / number_runs ;
% iter_aaj_average = iter_aaj_average / number_runs ;
iter_aaj2_average = iter_aaj2_average / number_runs ;
iter_pcg_average = iter_pcg_average / number_runs ;
% 
% display(['Richardson took', ' ', num2str(time_richardson), ' seconds', ' with error ', num2str(error_richardson)]);
% display(['Anderson took', ' ', num2str(time_aaj), ' seconds', ' with error ', num2str(error_aaj)]);
display(['Modified Anderson took', ' ', num2str(time_aaj2), ' seconds', ' with error ', num2str(error_aaj2)]);
display(['PCG took', ' ', num2str(time_pcg), ' seconds', ' with error ', num2str(error_pcg)]);

save(strcat('./results/', problem, preconditioner, '_preconditioner', '_',num2str(omega), '.mat'));
