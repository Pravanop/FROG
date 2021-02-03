# FROG

* The fftshift function was not working on our systems so we used fft_utils instead. 

In the With - pi by 2 phase.ipynb file, the first 9 cells are the same as in the original code. 

In the 10th cell, we have defined the function grads(T,i0, A, f0,a, a2, a3, f1, tau1, tau2) which computes derivatives of delta 
(the value of which needs to be minimised) with respect to all parameters. 

In the 11th cell, the function definition of def gradient_d(T, i0, A, f0, a, a2, a3, f1, tau1, tau2) is given.
This cell is where the gradient descent optimisation algorithm is implemented. There are some if statements we have included so as to avoid
values of the frequency becoming negative. In case f or T becomes negative (which isnt feasible), we decrease the learning rate and update the values accordingly.

In the 12th cell, we initialised the values of T, tau1, tau2 and iterated over the range provided to find a suitable starting point. Once T, tau1, tau2 are initialised,
we call the function gradient_d() which returns the list of parameters.

In the With plus pi by 2 phase.ipynb file, the first 9 cells are the same as in the original code, we have only changed the learning rates for the different parameters
for a quick convergence.

In the FROG_GD program, we have assumed T1 and T2 to be independent. Only the learning rates are different.

