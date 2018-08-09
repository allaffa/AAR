function [sol, count, rel_residual]=Richardson(A, rhs, omega, initial_guess, eps, iter, M1, M2)

    sol=initial_guess;

    rhs = M1 \ rhs;
    rhs = M2 \ rhs;
    
    vec = A * sol;
    vec = M1 \ vec;
    vec = M2 \ vec;
    
    rel_residual=norm(rhs-vec,2)/norm(rhs,2);
    count=0;

    r=rhs-vec;
    R=rel_residual;
    
    while(rel_residual>eps && count<=iter)
        
             count = count+1;
             sol = sol+omega*r; 
             
             vec = A * sol;
             vec = M1 \ vec;
             vec = M2 \ vec;
             
             r = rhs - vec;
             
             rel_residual = norm(r,2)/norm(rhs,2);
             R = [R rel_residual];    
             
    end


end
