clear 
close all
restoredefaultpath

maxNumCompThreads(1);

problem = 'e20r0000';
directory = strcat('./tests/', problem);
preconditioner = 'ilutp';

addpath(directory);

load('A.mat')

tol= 10^(-8);
omega = 0.2;
maxit = 100000;
restart = 6;
reorder = 0;
r = [];
Perm_ilutp = speye(size(A));
Perm_rcm = speye(size(A));

number_runs = 1;

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
    
    A = diag(diag(A))\A;
    
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
    
    [M1,M2]=ilu(D,setup); 
    
end


time_aaj2 = 0.0;
time_gmres2 = 0.0;

error_aaj2 = 0.0;
error_gmres2 = 0.0;

x = rand(size(A,1),1);

b = A * x;

x_guess = rand(size(A,1),1);

start_aaj2 = cputime;
% [x_aaj2, iter_aaj2, relres_aaj2, SOL_AAJ2] = AAJ2_correct(A,b,x_guess,tol,maxit,M1,M2,omega,1, restart+4,restart);
[x_aaj2, iter_aaj2, relres_aaj2, SOL_AAJ2] = AAJ2_augmented(A,b,x_guess,tol,maxit,M1,M2,omega,1,restart+4,restart);
finish_aaj2 = cputime;    

start_gmres2 = cputime;
% [ x_gmres2, relres_gmres2, iter_gmres2, ~, SOL_GMRES2 ] = gmres_prec ( A, x_guess, b, M1, M2, restart, maxit, tol);
finish_gmres2 = cputime;   

time_aaj2       = time_aaj2 + (finish_aaj2 - start_aaj2);
% time_gmres2     = time_gmres2 + (finish_gmres2 - start_gmres2);

error_aaj2       = error_aaj2 + norm(x_aaj2 - x)/norm(x);
% error_gmres2      = error_gmres2 + norm(x_gmres2 - x)/norm(x);

display(['Modified Anderson took', ' ', num2str(time_aaj2), ' seconds', ' with error ', num2str(error_aaj2)]);
% display(['GMRES2 took', ' ', num2str(time_gmres2), ' seconds', ' with error ', num2str(error_gmres2)]);

% figure()
% loglog(relres_aaj2, 'LineWidth', 3);
% hold on
% loglog(relres_gmres2, 'LineWidth', 3);
% legend('AAR', 'GMRES-Givens');
% xlabel('Iteration index');
% ylabel('Residual norm');
% title('Residual curve')
% set(gca, 'fontsize', 8)


%% Computation of the sequential angles

sequential_angles_aaj2 = [];
i = 1;
while i <= iter_aaj2 - 1 -restart
    res_prev = M2\(M1\(b - A*SOL_AAJ2(:,i)));
    res_new  = M2\(M1\(b - A*SOL_AAJ2(:,i+restart)));
    cosine   = abs(dot(res_prev,res_new))/(norm(res_prev)*norm(res_new));
    sequential_angles_aaj2 = [ sequential_angles_aaj2; acosd(cosine)];
    i = i + restart ;
end

% sequential_angles_gmres2 = [];
% i = 1;
% while i <= iter_gmres2(1) - 1
%     res_prev = M2\(M1\(b - A*SOL_GMRES2(:,i)));
%     res_new  = M2\(M1\(b - A*SOL_GMRES2(:,i+1)));
%     cosine   = abs(dot(res_prev,res_new))/(norm(res_prev)*norm(res_new));
%     sequential_angles_gmres2 = [ sequential_angles_gmres2; acosd(cosine)];
%     i = i + 1;
% end


% figure()
% semilogx([1:length(sequential_angles_aaj2)], sequential_angles_aaj2, 'LineWidth', 3)
% hold on
% semilogx([1:length(sequential_angles_gmres2)], sequential_angles_gmres2, 'LineWidth', 3)
% legend('AAR', 'GMRES-Givens')
% xlabel('Iteration index');
% ylabel('Angle between r_i and r_{i+1}');
% title('Sequential angles between r_i and r_{i+1} ')
% set(gca, 'fontsize', 8)


mean_sequential_angle_aaj2 = mean(sequential_angles_aaj2);
% mean_sequential_angle_gmres2 = mean(sequential_angles_gmres2);


%% Computation of the skip angles


skip_angles_aaj2 = [];
i = 1;
while i <= iter_aaj2 - 2 * restart
    res_prev = M2\(M1\(b - A*SOL_AAJ2(:,i)));
    res_new  = M2\(M1\(b - A*SOL_AAJ2(:,i+2*restart)));
    cosine   = abs(dot(res_prev,res_new))/(norm(res_prev)*norm(res_new));
    skip_angles_aaj2 = [ skip_angles_aaj2; acosd(cosine)];
    i = i + 2 * restart ;
end

% skip_angles_gmres2 = [];
% i = 1;
% while i <= iter_gmres2(1) - 2
%     res_prev = M2\(M1\(b - A*SOL_GMRES2(:,i)));
%     res_new  = M2\(M1\(b - A*SOL_GMRES2(:,i+2)));
%     cosine   = abs(dot(res_prev,res_new))/(norm(res_prev)*norm(res_new));
%     skip_angles_gmres2 = [ skip_angles_gmres2; acosd(cosine)];
%     i = i + 2;
% end


% figure()
% semilogx([1:length(skip_angles_aaj2)], skip_angles_aaj2, 'LineWidth', 3)
% hold on
% semilogx([1:length(skip_angles_gmres2)], skip_angles_gmres2, 'LineWidth', 3)
% legend('AAR', 'GMRES-Givens')
% xlabel('Iteration index');
% ylabel('Angle between r_i and r_{i+1}');
% title('Skip angles between r_{i-1} and r_{i+1} ')
% set(gca, 'fontsize', 8)
% 

mean_skip_angle_aaj2 = mean(skip_angles_aaj2);
% mean_skip_angle_gmres2 = mean(skip_angles_gmres2);


save(strcat('./results/angles/', problem, preconditioner, '_preconditioner', '.mat'));



