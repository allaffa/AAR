function [x_new, rango] = AndersonMixing_AAJ(x1, f1, m, iter, mix, p)
% f = residual, m = history, x=iterate, f(x)=g(x)-x, p=AJ step
% Reference:
%         Pratapa, Phanisri P., Phanish Suryanarayana, and John E. Pask. 
%         "Anderson acceleration of the Jacobi iterative method: 
%         An efficient alternative to Krylov methods for large, sparse linear systems." 
%         Journal of Computational Physics 306 (2016): 43-54.

persistent DX DF Xold Fold 
dp = 0;
rango =0;

if iter > 1

    if(size(DX,2)<m)
        DX = [DX, x1 - Xold];
        DF = [DF, f1 - Fold];
    else
        DX = [DX(:,2:end), x1 - Xold];
        DF = [DF(:,2:end), f1 - Fold];
    end

    if rem(iter,p)==0        % apply every p iters       
        
%         Yk = pinv(DF'*DF)*(DF'*f1);   
%         Yk = pinv(DF)*f1;  
        Yk = DF\f1;
        
        dp = (DX + mix*DF)*Yk;
    else
        dp=0;
    end    
end
x_new = x1 + mix*f1 - dp;    
    
Xold = x1;
Fold = f1;

