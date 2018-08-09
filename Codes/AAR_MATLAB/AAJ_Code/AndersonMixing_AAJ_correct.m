function [x_new] = AndersonMixing_AAJ_correct(x1, f1, m, iter, mix, p)
% f = residual, m = history, x=iterate, f(x)=g(x)-x, p=AJ step
% Reference:
%         Pratapa, Phanisri P., Phanish Suryanarayana, and John E. Pask. 
%         "Anderson acceleration of the Jacobi iterative method: 
%         An efficient alternative to Krylov methods for large, sparse linear systems." 
%         Journal of Computational Physics 306 (2016): 43-54.
persistent DX DF Xold Fold 
dp = 0;


if iter == 1
    
    x_new = x1 + mix * f1;

else
    k = mod(iter-2,m)+1;
    
    DX(:,k) = x1 - Xold;
    DF(:,k) = f1 - Fold;
    
    if rem(iter,p)==0        % apply every p iters  
        
%         Yk = pinv(DF'*DF)*(DF'*f1);        
        Yk = DF\f1;
        
        dp = DX*Yk;
        
        x_new = x1 - dp;
    else
        x_new = x1 + mix * f1;
    end    
end

Xold = x1;
Fold = f1;


