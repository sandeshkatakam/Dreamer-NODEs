
import abc
import jax 
import equinox as eqx
import jax.numpy as jnp
import jax.random as random
import 

_all_ = ['Abstract_NeuralODE', 'Neural_ODE']

class Abstract_NeuralODE(abc.ABC):
        def __init__(self, solver, t, y0, key, tol):
            self.solver = solver
            self.t = t
            self.y0 = y0
            self.tol = tol
            self.key = random.PRNGKey(0)


        def forward(self, func, t, y0):
            pass



    class Neural_ODE(Abstract_NeuralODE, eqx.Module):
        def __init__(self, solver, t, y0):
            super().__init__(solver, t, y0)

        def forward(self, func, t, y0):
            pass
