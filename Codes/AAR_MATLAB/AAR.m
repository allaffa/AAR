function [x] = AAR(A,b,x_guess,tol,max_iter,beta,m,p,PC)
% Preconditioned Alternating Anderson-Richardson (AAR) MATLAB code
% Copyright (C) 2017 Material Physics & Mechanics Group at Georgia Tech.
% Authors: Phanisri Pradeep Pratapa, Phanish Suryanarayana
% Last Modified: 22 July 2017
% Solves the system Ax = b
% Inputs: A       : square matrix (size N x N),
%         b       : right hand side column vector (size N x 1),
%         x_guess : initial guess (of size b),
%         tol     : convergence tolerance
%         max_iter: maximum number of iterations
%         beta    : extrapolation parameter
%         m       : Anderson history, no. of previous iterations to be considered in extrapolation
%         p       : Perform Anderson extrapolation at every p th iteration
%         PC      : Preconditioner, PC=0 (none), 1 (Jacobi) or 2 (ILU)
% Output: x       : solution vector
% NOTE  : A and b are to be provided. Other input parameters can be passed as "[]", to use defaults.
% e.g.  : Run code as, tic;x=AAR(A,b,[],[],[],[],[],[],[]);toc;

N=length(b);
%%%%%%%%%%%% Default Input Parameters %%%%%%%%%%%%
if isempty(x_guess)
    x_guess = ones(N,1);  % initial guess vector
end
if isempty(tol)
    tol = 1e-8;               % convergence tolerance
end
if isempty(max_iter)
    max_iter = 1e5;           % maximum number of iterations
end
if isempty(beta)
    beta = 0.2;               % extrapolation parameter
end
if isempty(m)
    m = 10;                   % Anderson history, no. of previous iterations to be considered in extrapolation
end
if isempty(p)
    p = 6;                    % Perform Anderson extrapolation at every p th iteration
end
if isempty(PC)
    PC = 0;                   % No preconditioning
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if PC==1
    D=diag(A); % Jacobi PC (AAJ)
    if nnz(D==0)~=0
        error('Error: Diagonal has zeros. Use a different PC.')
    end
elseif PC==2
    [L,U] = ilu(A,struct('type','nofill')); % ILU PC
end

nb = norm(b);
if nb==0
    nb=1;
    disp('Message: The RHS is zero. Calculating residual instead of relative residual.')
end

DX=zeros(N,m);
DF=zeros(N,m);

x_prev = x_guess;
relres = tol+1;
count = 1;
while count<=max_iter && relres > tol
    res = (b-A*x_prev);
    relres = norm(res)/nb;  % relative residual
    
    %fprintf('Iteration: %d, Relative Residual= %g \n',count,relres)
    
    % APPLY PC
    if PC==1
        res=res./D; % Jacobi PC (AAJ)
    elseif PC==2
        res = U\(L\(res)); % ILU PC
    end
    
    x_new = x_prev + beta*res; % Weighted Jacobi update
    
    % STORE HISTORY
    if count>1
        k = mod(count-2,m)+1;
        DX(:,k) = x_prev-Xold;
        DF(:,k) = res-Fold;
    end
    Xold = x_prev;
    Fold = res;
    
    % ANDERSON UPDATE
    if rem(count,p)==0  && count>1      % apply every p iters
        Yk = pinv(DF'*DF)*(DF'*res);
        dp = (DX + beta*DF)*Yk;
        x_new = x_new - dp;
    end
    
    x = x_prev;
    x_prev = x_new;
    count = count + 1;
end

if count-1 == max_iter
    fprintf('AAR exceeded maximum iterations and converged to a relative residual of %g. \n',relres);
else
    fprintf('AAR converged to a relative residual of %g in %d iterations.\n',relres,count-1);
end

end
