import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

n, m = 2, 3
x = cp.Variable(n)
A = cp.Parameter((m, n))
b = cp.Parameter(m)
c = cp.Parameter(m)
b_c = cp.Parameter(m)

constraints = [x >= 0]
constraints.append( A @ x >= b_c)
constraints.append( A @ x >= c)

objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))

problem = cp.Problem(objective, constraints)
assert problem.is_dpp()

cvxpylayer = CvxpyLayer(problem, parameters=[A, b, c, b_c], variables=[x])

A_tch = torch.randn(m, n, requires_grad=True)
b_tch = torch.randn(m, requires_grad=True)
c_tch = torch.randn(m, requires_grad=True)
b_times_c = torch.multiply(b_tch, c_tch)
print(b_times_c)

# solve the problem
solution, = cvxpylayer(A_tch, b_tch, c_tch, b_times_c)

# compute the gradient of the sum of the solution with respect to A, b
solution.sum().backward()