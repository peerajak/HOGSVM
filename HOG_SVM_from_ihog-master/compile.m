% Run this script to compile the entire system.

fprintf('compiling features.cc\n');
mex -O internal/features.cc -output internal/features
mex qp.cc;       % QP solver for SVM
mex addcols.cc
mex resize.cc
fprintf('compiling spams\n');
run 'spams/compile.m'
