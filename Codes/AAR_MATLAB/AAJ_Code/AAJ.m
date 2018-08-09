function [x, iter, reshist, SOL_HIST] = AAJ(A,b,x_guess,tol, max_iter, beta, m,p)

% Alternating Anderson-Jacobi (AAJ) method
% Jacobi method accelerated using Anderson mixing (every p iterations)
% Solves the system Ax = b
% Inputs: A       : matrix, 
%         b       : right hand side vector, 
%         x_guess : initial guess,
%         tol     : convergence tolerance
% Output: solution vector x
%         iter: number of iterations taken to converge
%         reshist: history of the relative residual norm
%         SOL_HIST: columns are made of solution vectors at each iteration.
% Running the code:
%         Load the matrix file that has a matrix (A) and a right hand
%         side vector b. Then run the code as:
%         tic; x = AAJ(A,b,x_guess,tol); toc;
% Modify the "Input Parameters" specified below, as required
% Reference:
%         Pratapa, Phanisri P., Phanish Suryanarayana, and John E. Pask. 
%         "Anderson acceleration of the Jacobi iterative method: 
%         An efficient alternative to Krylov methods for large, sparse linear systems." 
%         Journal of Computational Physics 306 (2016): 43-54.

%tol = 1e-8;              % convergence tolerance
%x_guess = ones(size(b)); % initial guess

%%%%%%%%%%%% Input Parameters %%%%%%%%%%%%
% max_iter = 1e6;           % maximum number of iterations 
% beta = 0.2;               % mixing parameter
% m = 10;                   % Anderson history, no. of previous iterations to be considered in mixing
% p = 6;                    % Perform Anderson mixing at every p th iteration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A = D + R
D = diag(A);
R = A-diag(D);
if nnz(D==0)~=0
    error('Error: Diagonal has zeros. Use a different solver.')
end
q = b./D;
nq = norm(q);
if nq==0
    nq=1;
    disp('Message: The RHS is zero. Calculating residual instead of relative residual.')
end

x_prev = x_guess;

clear AndersonMixing_AAJ
relres = tol+1;
count = 1;

SOL_HIST=[];

SOL_HIST = [SOL_HIST x_prev];

while count<=max_iter && relres > tol
    x_new = q - (R*x_prev)./D;    
    SOL_HIST = [SOL_HIST x_new];
    relres = norm(x_new-x_prev)/nq; % relative residual
    reshist(count) = relres;
%     fprintf('Iteration: %d, Residual= %g \n',count,relres)
    x = x_new;    
    x_new = AndersonMixing_AAJ(x_prev,  x_new-x_prev,  m,  count,  beta, p);    
    x_prev = x_new;
    count = count + 1;
end

clear AndersonMixing_AAJ

iter = count-1;

if count-1 == max_iter
    fprintf('Alternating Anderson-Jacobi (AAJ) exceeded maximum iterations and converged to a tolerance of %g. \n',relres);
else
    fprintf('Alternating Anderson-Jacobi (AAJ) converged to a tolerance of %g, in %d iterations.\n',relres,count-1);
end

end
