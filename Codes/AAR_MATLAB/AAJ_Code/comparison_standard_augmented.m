clear 
close all
restoredefaultpath

maxNumCompThreads(1);

problem = 'sherman3';
directory = strcat('./tests/', problem);
preconditioner = 'none';

addpath(directory);

load('A.mat')

tol= 10^(-8);
maxit = 10^6;

number_runs = 10;

if strcmp(preconditioner, 'none')
    
    M1 = speye(size(A));
    M2 = speye(size(A));
    
elseif strcmp(preconditioner, 'diagonal')
    
    M1 = speye(size(A));
    M2 = diag(diag(A));
    
elseif strcmp(preconditioner, 'block')
    
    size_block = 43;
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

    [M1,M2]=ilu(A,setup);
    
 elseif  strcmp(preconditioner, 'ilutp')    
    
    setup.type='ilutp';
    setup.droptol = 10^(-3);
    setup.udiag = 1;
    setup.thresh = 0.9;
    
    [M1,M2]=ilu(A,setup);
    
elseif strcmp(preconditioner, 'block_ilu')    
    
    size_block = 5;
    n_blocks = size(A,1)/size_block;
    D = sparse(size(A));

    for i = 0:n_blocks-1
        D(i*size_block+1:(i+1)*size_block,i*size_block+1:(i+1)*size_block) = A(i*size_block+1:(i+1)*size_block,i*size_block+1:(i+1)*size_block);
    end

    D = sparse(D);
    
    setup.type='ilutp';
    setup.droptol = 10^(-4);
    setup.udiag = 1;
    
    [M1,M2]=ilu(D,setup); 
    
end


time_aaj2 = 0.0;
error_aaj2 = 0.0;
iter_aaj2_average = 0.0;

time_aaj2_augmented = 0.0;
error_aaj2_augmented = 0.0;
iter_aaj2_augmented_average = 0.0;


for runs = 1:number_runs
    
    x = rand(size(A,1),1);

    b = A * x;
    
    x_guess = rand(size(A,1),1);
    
    start_aaj2 = cputime;
    [x_aaj2, iter_aaj2, relres_aaj2] = AAJ2(A,b,x_guess,tol,maxit,M1,M2,2/norm(A,'inf'),1,12,6);
    finish_aaj2 = cputime;    

    start_aaj2_augmented = cputime;
    [x_aaj2_augmented, iter_aaj2_augmented, relres_aaj2_augmented] = AAJ2_augmented(A,b,x_guess,tol,maxit,M1,M2,2/norm(A,'inf'),1,12,6);
    finish_aaj2_augmented = cputime;    

    time_aaj2         = time_aaj2 + (finish_aaj2 - start_aaj2);
    error_aaj2        = error_aaj2 + norm(x_aaj2 - x)/norm(x);
    iter_aaj2_average = iter_aaj2_average + iter_aaj2;

    time_aaj2_augmented         = time_aaj2_augmented + (finish_aaj2_augmented - start_aaj2_augmented);
    error_aaj2_augmented        = error_aaj2_augmented + norm(x_aaj2_augmented - x)/norm(x);
    iter_aaj2_augmented_average = iter_aaj2_augmented_average + iter_aaj2_augmented;


end

time_aaj2                 = time_aaj2 / number_runs;
time_aaj2_augmented       = time_aaj2_augmented / number_runs;

% 
error_aaj2                 = error_aaj2 / number_runs;
error_aaj2_augmented       = error_aaj2_augmented / number_runs;

% 
iter_aaj2_average           = iter_aaj2_average / number_runs ;
iter_aaj2_augmented_average = iter_aaj2_augmented_average / number_runs ;

% 
display(['Modified Anderson took', ' ', num2str(time_aaj2), ' seconds', ' with error ', num2str(error_aaj2)]);
display(['Modified Augmented Anderson took', ' ', num2str(time_aaj2_augmented), ' seconds', ' with error ', num2str(error_aaj2_augmented)]);

save(strcat('./results/', problem, preconditioner, '_preconditioner', '.mat'));

