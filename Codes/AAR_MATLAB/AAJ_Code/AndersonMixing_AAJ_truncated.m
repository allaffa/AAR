function [x_new] = AndersonMixing_AAJ_truncated(x1, f1, iter, mix, trunc)
% f = residual, m = history, x=iterate, f(x)=g(x)-x, p=AJ step
% Reference:
%         Pratapa, Phanisri P., Phanish Suryanarayana, and John E. Pask. 
%         "Anderson acceleration of the Jacobi iterative method: 
%         An efficient alternative to Krylov methods for large, sparse linear systems." 
%         Journal of Computational Physics 306 (2016): 43-54.

persistent DX DF Xold Fold 
dp = 0;

if iter > 1
    
        if size(DF,2)<trunc
            
            DF = [DF, f1-Fold];
            DX = [DX, x1-Xold];   
              
        else
            
            DF = [DF(:,2:end), f1-Fold];
            DX = [DX(:,2:end), x1-Xold];
            
        end
   

        Yk = DF\f1;
        dp = (DX + mix*DF)*Yk;

end

x_new = x1 + mix*f1 - dp;

Xold = x1;
Fold = f1;