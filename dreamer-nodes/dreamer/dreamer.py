import jax
import jax.numpy as jnp
import equinox as eqx
from equinox import nn as Module
import diffrax
import optax


## Notes 
'''
1. Implement Base Dreamer Class
2. Implement Dreamer V1, Dreamer V2, Dreamer V3 as Sub_Classes of Dreamer
3. Implement Actor and Critic Networks part of the Dreamer Architecture
4. Try finding out solution on how to implement the LatentODE class in the Dreamer Architecture
5. Use Equinox, Diffrax and JAX libraries

'''
class Base_Dreamer(eqx.Module):
    def __init__(self, *, data_size, hidden_size, latent_size, width_size, depth, key, **kwargs):
        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.width_size = width_size
        self.depth = depth
        self.key = key
    
    def __call__(self):
        pass

    def __iter__(self):
        pass



class DreamerV1(Base_Dreamer):
    def __init__(self, *, data_size, hidden_size, latent_size, width_size, depth, key, **kwargs):
        super().__init__(data_size, hidden_size, latent_size, width_size, depth, key, **kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.width_size = width_size
        self.depth = depth
        self.key = key
    
    def __call__(self):
        pass

    def __iter__(self):
        pass


class DreamerV2(Base_Dreamer):
    def __init__(self, *, data_size, hidden_size, latent_size, width_size, depth, key, **kwargs):
        super().__init__(data_size, hidden_size, latent_size, width_size, depth, key, **kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.width_size = width_size
        self.depth = depth
        self.key = key
    
    def __call__(self):
        pass

    def __iter__(self):
        pass


class DreamerV3(Base_Dreamer):
    def __init__(self, *, data_size, hidden_size, latent_size, width_size, depth, key, **kwargs):
        super().__init__(data_size, hidden_size, latent_size, width_size, depth, key, **kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.width_size = width_size
        self.depth = depth
        self.key = key
    
    def __call__(self):
        pass

    def __iter__(self):
        pass

    