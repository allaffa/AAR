function [A, b] = block_circulant(m)
 
    b = zeros(m*(m+1),1);
    x = b;
    A = [];
    for k = 1:m
        C = eye(2*k,2*k);
        C = [C(:,2:2*k),C(:,1)];
        A = blkdiag(A,C);
        b(k*(k-1)+1,1) = 1;
    end

end