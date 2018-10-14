function [A, b] = block_circulant(l,m)
 
    b = zeros(m*(m+1),1);
    x = b;
    A = [];
    for k = 1:m
        C = eye(l*k,l*k);
        C = [C(:,2:l*k),C(:,1)];
        A = blkdiag(A,C);
        b(k*(k+1)/2*l,1) = 1;
    end

end