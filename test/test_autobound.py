import autobound.jax as ab
import numpy as np
import jax.numpy as jnp
import torch

from dynamic_system_instances import inverted_pendulum_1

# f = lambda x: x**2 # 1.5*jnp.exp(3*x) - 25*x**2
L = 1
b = 0.1
m = 1.0


def f(s: torch.Tensor) -> torch.Tensor:
    
    return torch.prod(s)


# f = lambda x: 1.5*jnp.sin(3*x) - 25*x**2

x0 = np.zeros((1, inverted_pendulum_1.ns)) 
interval = np.ones((1, inverted_pendulum_1.ns))
trust_region = (x0- interval, x0 + interval)

# Compute quadratic upper and lower bounds on f.
bounds = ab.taylor_bounds(f, max_degree=0)(x0, trust_region)
print(bounds.lower(0), bounds.upper(1))

# print(inverted_pendulum_1.control_limits)