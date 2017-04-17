function [w,cache] = qpopt(x,n,C,TolRel)
% [w,cache] = qpopt(x,n,C,TolRel)
%
% Solves the following qp via a cutting plane algoritm
%                     min .5*w*w + C*sum_i Ei
%  with constraint  w*x_i > n_i - Ei
%  
% 1) Assumes all data points are 'positive', so use -x_i for negative examples 
%
% 2) Usually n_i = 1
%    But this general formulation can naturally handle weighted slacks
%    eg, weighting the ith slack by 's_i' amounts to using the constraint:
%    w*s_i*x_i > s_i - Ei

VERBOSE = 1;  

if VERBOSE > 0,
  fprintf('\nIn QPopt');
end

assert(isa(x,'single'));
assert(isa(n,'single'));

xc = [];
nc = [];
a = [];
H = [];
w = zeros(size(x,1),1);

slack = n-w'*x;
err   = slack > 0;
loss  = sum(slack(err));
lb    = 0;
ub    = w'*w*.5 + C*loss;
w_best= w; 
a_best= a;
err_p = zeros(size(err));

t     = 1;
tmax = 1000; %  1000;

% Repeat while
% 1) upper and lower bounds are too far apart
% 2) new constraints are being added
% 3) we haven't hit max iteration count
while 1 - lb/ub > TolRel && any(err_p ~= err) && t < tmax,

  % Compute new constraint
  I  = find(err);
  xi = addcols(x,I);
  ni = addcols(n,I);
  if isempty(xc),
    Hi = [];
  else
    Hi = xi'*xc;
  end

  % Add constraint to cache
  xc = [xc xi];
  nc = [nc ni];
  H  = [H Hi';Hi xi'*xi];
  a  = [a; 0];
  
  % Store active examples
  sv{length(a)} = I;
  
  % Call qp solver to solve dual
  I = ones(size(a),'uint32');
  S = ones(size(a),'uint8');
  [a,v] = qp(H,-nc,C,I,S,a,inf,0,TolRel,-inf,0);
  
  % Update lower bound
  lb = -v;

  % Find new constraint to add to cache
  w = xc*a;
  slack = n-w'*x;
  err  = slack > 0;
  loss = sum(slack(err));
  obj   = w'*w*.5 + C*loss;

  % Update upper bound
  if obj < ub,
    ub     = obj;
    w_best = w;
    a_best = a;
  end

  switch VERBOSE
    case 2
     svs = logical(zeros(size(slack)));
     I = find(a > 0);
     for i = I',
       svs(sv{i}) = 1;
     end
      fprintf('\n#planes=%d,#sv=%d,lb=%.3g,ub=%.3g',length(a),sum(svs),lb,ub);
    case 1      
      fprintf('.');
  end

  t = t + 1;
end

if t >= tmax,
  %fprintf('\nQPopt did not converge; cost=%g,cost_cache=%g',cost,cost_cache);    
  fprintf('\nQPopt did not converge; cost=%g',obj);
end

% Return back active set and support vectors
a = a_best;
w = w_best;
I = find(a > 0);
cache.xc = xc(:,I);
cache.nc = nc(I);
cache.a  = a(I);
cache.ub = ub;
cache.lb = lb;
cache.iter  = t;

% Collect all points which are included in the active cutting planes
cache.sv = logical(zeros(size(slack)));
for i = I',
  cache.sv(sv{i}) = 1;
end

if VERBOSE > 0,
  fprintf('#planes=%d,#sv=%d,lb=%.3g,ub=%.3g',length(I),sum(cache.sv),lb,ub);
end

return;
