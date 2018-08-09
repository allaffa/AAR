clear 
close all
restoredefaultpath

maxNumCompThreads(1);

problem = 'garon1';

directory = strcat('./tests/', problem);
preconditioner = 'ilutp';

addpath(directory);

load('A.mat')

tol= 10^(-8);
omega = 0.2;
beta = 1;
maxit = 100000;
restart = 10;
reorder = 0;
r = [];
Perm_ilutp = speye(size(A));
Perm_rcm = speye(size(A));

number_runs = 10;

if strcmp(preconditioner, 'none')
    
    M1 = speye(size(A));
    M2 = speye(size(A));
    
elseif strcmp(preconditioner, 'diagonal')
    
    M1 = speye(size(A));
    diag_vector = diag(A);
    zero_positions = find(diag_vector == 0);
    diag_vector(zero_positions) = 1.0;
    M2 = diag(diag_vector);
    
elseif strcmp(preconditioner, 'block')
    
    size_block = 8;
    n_blocks = size(A,1)/size_block;

    D_inv = sparse(size(A));
    D = sparse(size(A));

    for i = 0:n_blocks-1
        D(i*size_block+1:(i+1)*size_block,i*size_block+1:(i+1)*size_block) = A(i*size_block+1:(i+1)*size_block,i*size_block+1:(i+1)*size_block);
        D_inv(i*size_block+1:(i+1)*size_block,i*size_block+1:(i+1)*size_block) = inv(A(i*size_block+1:(i+1)*size_block,i*size_block+1:(i+1)*size_block));
    end

    D = sparse(D);
    
    M1 = speye(size(A));
    M2 = D;

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
    
  elseif  strcmp(preconditioner, 'ilu0')    
    
    setup.type='nofill';
    setup.milu='off';
    
    if (reorder ==1)
        r = symrcm(A);
        Perm_rcm = speye(size(A));
        Perm_rcm = Perm_rcm(r,r);
        A = A(r,r);
    end
       
    [M1,M2]=ilu(A,setup);
    
 elseif  strcmp(preconditioner, 'ilutp')    
    
    setup.type='ilutp';
    setup.droptol = 10^(-4);
    setup.udiag = 1;
    setup.thresh = 0;
        
    if (reorder ==1)
        r = symrcm(A);
        Perm_rcm = speye(size(A));
        Perm_rcm = Perm_rcm(r,r);
        A = A(r,r);
    end
        
    [M1,M2, Perm_ilutp]=ilu(A,setup);
    
    assert( sum(sum(Perm_ilutp~=speye(size(A,1))))==0 );
    
elseif strcmp(preconditioner, 'block_ilu')    
    
    size_block = 8;
    n_blocks = size(A,1)/size_block;
    D = sparse(size(A));

    for i = 0:n_blocks-1
        D(i*size_block+1:(i+1)*size_block,i*size_block+1:(i+1)*size_block) = A(i*size_block+1:(i+1)*size_block,i*size_block+1:(i+1)*size_block);
    end

    D = sparse(D);
    
    setup.type='ilutp';
    setup.droptol = 10^(-4);
    setup.udiag = 1;
    
    A = diag(diag(A))\A;
    
    [M1,M2]=ilu(D,setup); 
    
end

fprintf('CONSTRUCTION OF THE PRECONDITIONER FINISHED \n\n');

time_richardson = 0.0;
time_aaj = 0.0;
time_aaj2 = 0.0;
time_gmres = 0.0;
time_gmres2 = 0.0;

error_richardson = 0.0;
error_aaj = 0.0;
error_aaj2 = 0.0;
error_gmres = 0.0;
error_gmres2 = 0.0;

iter_richardson_average = 0.0;
iter_aaj_average = 0.0;
iter_aaj2_average = 0.0;
iter_gmres_average = 0.0; 
iter_gmres2_average = 0.0;


for runs = 1:number_runs
    
%     x = rand(size(A,1),1);
    x = ones(size(A,1),1);

    b = A * x;
    
%     x_guess = rand(size(A,1),1);
      x_guess = zeros(size(A,1),1);
    
%     start_richardson = cputime;
%     [x_rich, iter_rich, relres_rich] = Richardson( A, b, 1, x_guess, tol, maxit, M1, M2 );
%     finish_richardson = cputime;

    start_aaj2 = cputime;
    [x_aaj2, iter_aaj2, relres_aaj2] = AAJ2(A,b,x_guess,tol,maxit,M1,M2,omega,beta,10,1);
%     [x_aaj2, iter_aaj2, relres_aaj2] = AAJ2_full(A,b,x_guess,tol,maxit,M1,M2,1);
%     [x_aaj2, iter_aaj2, relres_aaj2] = AAJ2_restart(A,b,x_guess,tol,maxit,M1,M2, 2/norm(A, 'inf'), 1, restart, 6, restart);
%     [x_aaj2, iter_aaj2, relres_aaj2] = AAJ2_truncated(A,b,x_guess,tol,maxit,M1,M2,1,10);
%     [x_aaj2, iter_aaj2, relres_aaj2] = AAJ2_correct(A,b,x_guess,tol,maxit,M1,M2,0.2,1,restart-1,restart);
%     [x_aaj2, iter_aaj2, relres_aaj2] = AAJ2_augmented(A,b,x_guess,tol,maxit,M1,M2,omega,beta,10,6);
    finish_aaj2 = cputime;    

%     start_richardson = cputime;
%     [x_rich, iter_rich, relres_rich] = Richardson_prec_inv( A, b, x_guess, tol, maxit, M2_inv, M1_inv, 10, 6 );
%     finish_richardson = cputime;
% 
%     start_aaj = cputime;
%     [x_aaj, iter_aaj] = AAJ(A,b,x_guess,tol,maxit, 10, 6);
%     finish_aaj = cputime;
  

%     start_gmres = cputime;
%     [x_gmres,flag_gmres,res_final,iter_gmres, relres_gmres] = gmres(A,b,restart,tol,maxit,M1,M2,x_guess);
%     finish_gmres = cputime;   


    start_gmres2 = cputime;
%    [ x_gmres2, relres_gmres2, iter_gmres2 ] = gmres_prec ( A, x_guess, b, M1, M2, restart, maxit, tol);
%    [ x_gmres2, relres_gmres2, iter_gmres2 ] = gmres_prec_trunc ( A, x_guess, b, M1, M2, maxit, 10, maxit, tol);
    finish_gmres2 = cputime;   
    
%     if strcmp(preconditioner, 'ilutp')  
%         x_aaj2 = Perm_ilutp * x_aaj2;
%         x_gmres2 = Perm_ilutp * x_gmres2;        
%     end
   
%     start_aaj2 = cputime;
%     [x_aaj2, iter_aaj2] = AAJ2_prec_inv(A,b,x_guess,tol,maxit,M2_inv,M1_inv, 1, 10, 6);
%     [x_aaj2, iter_aaj2] = AAJ2_restart_prec_inv(A,b,x_guess,tol,maxit,M2_inv,M1_inv, 1, restart, 1, restart);
%     finish_aaj2 = cputime;

%     start_gmres2 = cputime;
%     [ x_gmres2, relres_gmres2, iter_gmres2 ] = gmres_prec_inv ( A, x_guess, b, M2_inv, M1_inv, restart, maxit, tol);
%     finish_gmres2 = cputime;   
% 

%     time_richardson = time_richardson + (finish_richardson - start_richardson);
%     time_aaj        = time_aaj + (finish_aaj - start_aaj);
     time_aaj2       = time_aaj2 + (finish_aaj2 - start_aaj2);
%     time_gmres      = time_gmres + (finish_gmres - start_gmres);
%     time_gmres2      = time_gmres2 + (finish_gmres2 - start_gmres2);
%     
%     error_richardson = error_richardson + norm(x_rich - x)/norm(x);
%     error_aaj        = error_aaj + norm(x_aaj - x)/norm(x);
     error_aaj2       = error_aaj2 + norm(x_aaj2 - x)/norm(x);
%     error_gmres      = error_gmres + norm(x_gmres - x)/norm(x);
%     error_gmres2     = error_gmres2 + norm(x_gmres2 - x)/norm(x);
%     
%     iter_richardson_average = iter_richardson_average + iter_rich;
%     iter_aaj_average = iter_aaj_average + iter_aaj;
    iter_aaj2_average = iter_aaj2_average + iter_aaj2;
%     iter_gmres_average = iter_gmres_average + iter_gmres;
%     iter_gmres2_average = iter_gmres2_average + iter_gmres2;

end


% time_richardson = time_richardson / number_runs;
% time_aaj        = time_aaj / number_runs;
time_aaj2       = time_aaj2 / number_runs;
% time_gmres      = time_gmres / number_runs;
% time_gmres2     = time_gmres2 / number_runs;
% 
% error_richardson = error_richardson / number_runs;
% error_aaj        = error_aaj / number_runs;
error_aaj2       = error_aaj2 / number_runs;
% error_gmres      = error_gmres / number_runs;
% error_gmres2     = error_gmres2 / number_runs;
% 
% iter_richardson_average = iter_richardson_average / number_runs ;
% iter_aaj_average = iter_aaj_average / number_runs ;
iter_aaj2_average = iter_aaj2_average / number_runs ;
% iter_gmres_average = iter_gmres_average / number_runs ;
% iter_gmres2_average = iter_gmres2_average / number_runs ;
% 
% display(['Richardson took', ' ', num2str(time_richardson), ' seconds', ' with error ', num2str(error_richardson)]);
% display(['Anderson took', ' ', num2str(time_aaj), ' seconds', ' with error ', num2str(error_aaj)]);
display(['Modified Anderson took', ' ', num2str(time_aaj2), ' seconds', ' with error ', num2str(error_aaj2)]);
% display(['GMRES took', ' ', num2str(time_gmres), ' seconds', ' with error ', num2str(error_gmres)]);
% display(['GMRES2 took', ' ', num2str(time_gmres2), ' seconds', ' with error ', num2str(error_gmres2)]);


% loglog(relres_aaj2, 'LineWidth', 3);
% hold on
% loglog(relres_gmres2, 'LineWidth', 3);
% legend('AAR', 'GMRES-Givens')
% set(gca, 'fontsize', 8)


save(strcat('./results/', problem, preconditioner, '_preconditioner', '_',num2str(omega), '.mat'));
