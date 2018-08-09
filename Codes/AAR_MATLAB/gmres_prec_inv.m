function [ x, reshist, iter_vec, flag, SOL_HIST ] = gmres_prec_inv ( A, x, b, M1, M2, restart, max_it, tol)

%*****************************************************************************80
%
%% GMRES solves a linear system Ax=b using the Generalized Minimal residual.
%
%  Discussion:
%
%    The routine uses restarts; it does NOT include preconditioning.
%
%  Modified:
%
%    19 July 2004
%
%  Reference:
%
%    Richard Barrett, Michael Berry, Tony Chan, James Demmel,
%      June Donato, Jack Dongarra, Victor Eijkhout, Roidan Pozo,
%      Charles Romine, Henk van der Vorst
%    Templates for the Solution of Linear Systems: Building Blocks for 
%      Iterative Methods, 
%    SIAM Publications, 1993.
%
%  Parameters:
%
%    Input, real A(N,N), the nonsymmetric positive definite matrix.
%
%    Input, real X(N), the initial guess vector.
%
%    Input, real B(N), the right hand side vector.
%
%    Input, real M, not used.
%
%    Input, integer RESTART, the number of iterations between restarts.
%
%    Input, integer MAX_IT, the maximum number of iterations.
%
%    Input, real TOL, an error tolerance.
%
%    Input, integer N, the order of the matrix A.
%
%    Output, real X(N), the solution.
%
%    Output, real RES_NORM, the norm of the residual.
%
%    Output, integer ITER, the number of iterations performed.
%
%    Output, integer FLAG, the return flag.
%    0 = the solution was found to within the specified tolerance.
%    1 = a satisfactory solution was not found.  The iteration limit
%        was exceeded.
%
%    Output, matrix SOL_HIST, columns are made of solution vectors at each iteration.
%
  iter_vec= zeros(2,1);
  flag = 0;
  res_in = (M2 * b);
  res_in = (M1 * res_in);
  bnrm2 = norm(res_in);
  
  n = size(A,1);

  if ( bnrm2 == 0.0 )
    bnrm2 = 1.0;
  end

  r = ( M2*( b - A * x ) );
  r = M1 * r;
  
  res_norm = norm ( r ) / bnrm2;
  
  res_hist = res_norm;
 
  if ( res_norm < tol ) 
    return
  end

  m = restart;
  V = sparse(n,m+1);
  H = sparse(m+1,m);
  cs(1:m) = zeros(m,1);
  sn(1:m) = zeros(m,1);
  e1 = zeros(n,1);
  e1(1) = 1.0;

  iter2 = 0;
  
  SOL_HIST = [];
  
  SOL_HIST = [SOL_HIST x];

  for iter = 1: ceil(max_it / m) ;    % changed by L. Foster, this change
                                       % will correspond to one iteration
                                       % for every matrix-vector mult.
%%    r = M \ ( b-A*x );          % changed by L. Foster, done before
                                  %    each loop repetition
    V(:,1) = r / norm( r );
    s = norm ( r ) * e1;
%
%  Construct an orthonormal basis using Gram-Schmidt orthogonalization.
%
    iter_vec(1)=1;
    for i = 1 : m

      w = M2 * (A * V(:,i));
      w = M1 * w;

      for k = 1 : i
        H(k,i)= w' * V(:,k);
        w = w - H(k,i) * V(:,k);
      end

      H(i+1,i) = norm ( w );
      V(:,i+1) = w / H(i+1,i);
%
%  Apply the Givens rotation.
%
      for k = 1 : i-1
        temp     =  cs(k) * H(k,i) + sn(k) * H(k+1,i);
        H(k+1,i) = -sn(k) * H(k,i) + cs(k) * H(k+1,i);
        H(k,i)   = temp;
      end
%
%  Form the I-th rotation matrix.
%
      aa = H(i,i);
      bb = H(i+1,i);

      if ( bb == 0.0 )
        cs(i) = 1.0;
        sn(i) = 0.0;
      elseif ( abs ( aa ) < abs ( bb ) )
        temp = aa / bb;
        sn(i) = 1.0 / sqrt( 1.0 + temp^2 );
        cs(i) = temp * sn(i);
      else
        temp = bb / aa;
        cs(i) = 1.0 / sqrt( 1.0 + temp^2 );
        sn(i) = temp * cs(i);
      end
%
%  Approximate the residual norm.
%                  
      temp   = cs(i) * s(i);
      s(i+1) = -sn(i) * s(i);
      s(i)   = temp;
      H(i,i) = cs(i) * H(i,i) + sn(i) * H(i+1,i);
      H(i+1,i) = 0.0;
      res_norm  = abs ( s(i+1) ) / bnrm2;
%       iter2 = iter2 + 1;
      reshist = [reshist res_norm];
%
%  Update the approximate solution.
%
      if ( res_norm <= tol | max_it <= iter2 )
        y = H(1:i,1:i) \ s(1:i);
        x = x + V(:,1:i) * y;
        break;
      end
    end

    iter2 = iter2 + 1;
    iter_vec(2)=i;

    if ( res_norm <= tol | max_it <= iter2 )
      break;
    end

    y = H(1:m,1:m) \ s(1:m);
%
%  Update the approximation.
%
    x = x + V(:,1:m) * y;
%
%  Compute the residual.
%
    r =  M2 * ( b-A*x );
    r =  M1 * r;
    
    SOL_HIST = [SOL_HIST x];
                          
    s(i+1) = norm ( r );
    res_norm = s(i+1) / bnrm2;

    if ( res_norm <= tol | max_it <= iter2 )
      break;
    end

  end

  if ( tol < res_norm ) 
    flag = 1;
  end;

  iter_vec(1) = iter2;

  return
end
