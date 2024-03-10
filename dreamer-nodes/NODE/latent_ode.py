##############################
# Latent ODE Networks 
# From the Paper: https://arxiv.org/abs/1907.03907
# Authors: Yulia Rubanova, Ricky T. Q. Chen, David Duvenaud, Karol Gregor, and Michael A. Osborne
##############################



from base_neural_ode import Abstract_NeuralODE
import time

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax


matplotlib.rcParams.update({"font.size": 30})



_all_ = ['LatentODE']


class LatentODE(Abstract_NeuralODE):
    

    def __init__(self, solver, t, y0, key, tol):
        super().__init__(solver, t, y0, key, tol)
        self.solver = solver
        self.t = t
        self.y0 = y0
        self.key = jax.random.PRNGKey(0)
        self.tol = tol

    def _latent(self, ts, ys, key):
        data = jnp.concatenate([ts[:, None], ys], axis=1)
        hidden = jnp.zeros((self.hidden_size,))
        for data_i in reversed(data):
            hidden = self.rnn_cell(data_i, hidden)
        context = self.hidden_to_latent(hidden)
        mean, logstd = context[: self.latent_size], context[self.latent_size :]
        std = jnp.exp(logstd)
        latent = mean + jr.normal(key, (self.latent_size,)) * std
        return latent, mean, std

    # Decoder of the VAE
    def _sample(self, ts, latent):
        dt0 = 0.4  # selected as a reasonable choice for this problem
        y0 = self.latent_to_hidden(latent)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return jax.vmap(self.hidden_to_data)(sol.ys)

    def forward(self, func, t, y0):
        pass