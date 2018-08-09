function [x_new] = AndersonMixing_AAJ_augmented(x1, f1, m, iter, mix, p)
% f = residual, m = history, x=iterate, f(x)=g(x)-x, p=AJ step
% Reference:
%         Pratapa, Phanisri P., Phanish Suryanarayana, and John E. Pask. 
%         "Anderson acceleration of the Jacobi iterative method: 
%         An efficient alternative to Krylov methods for large, sparse linear systems." 
%         Journal of Computational Physics 306 (2016): 43-54.
persistent DX DF Xold Fold DXmixing DFmixing drop_first
dp = 0;

drop_first = 0;

if drop_first == 0
    m = m+1;
end

if iter > 1

    if(size(DX,2)<m)
        if rem(iter-1,p)~=0
            DX = [DX, x1 - Xold];
            DF = [DF, f1 - Fold];
        else
            DX = [DX, x1 - Xold + DXmixing, DXmixing];
            DF = [DF, f1 - Fold + DFmixing, DFmixing];            
        end
    else
        if rem(iter-1,p)~=0
            DX = [DX(:,2:end), x1 - Xold];
            DF = [DF(:,2:end), f1 - Fold];
        else
            if(drop_first==0)
                DX = [DX(:,3:end), x1 - Xold + DXmixing, DXmixing];
                DF = [DF(:,3:end), f1 - Fold + DFmixing, DFmixing];
            else
                DX = [DX(:,2:end), x1 - Xold + DXmixing];
                DF = [DF(:,2:end), f1 - Fold + DFmixing];
            end
        end
    end

    if rem(iter,p)==0        % apply every p iters       
        Yk = DF\f1;
        
        DXmixing = DX*Yk;
        DFmixing = DF*Yk;
        dp = DXmixing + mix*DFmixing;
        
        if(Yk(1)==0)
            drop_first = 1;
        else
            drop_first = 0;
        end
        
    else
        dp=0;
    end    
end
x_new = x1 + mix*f1 - dp;    
    
Xold = x1;
Fold = f1;
