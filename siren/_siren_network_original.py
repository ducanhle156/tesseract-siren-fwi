# utils/siren_network.py
"""SIREN neural network implementation"""

import jax
import jax.numpy as jnp
from jax import random, vmap
from .config import CONFIG

def initialize_networks(rng_key):
    """Initialize the SIREN networks"""
    params1 = init_mlp_params(CONFIG.network.neurons, rng_key)
    params2 = init_mlp_params(CONFIG.network.neurons, rng_key)
    return [params1, params2]

def init_mlp_params(layer_widths, rng_key):
    """
    Initialize parameters for SIREN network with PyTorch-style initialization.
    
    Args:
        layer_widths: List of layer widths [input_dim, hidden1, hidden2, ..., output_dim]
        rng_key: JAX random key
    
    Returns:
        List of [weight, bias] pairs for each layer
    """
    params = []
    keys = random.split(rng_key, len(layer_widths) - 1)
    
    for i, (n_in, n_out) in enumerate(zip(layer_widths[:-1], layer_widths[1:])):
        weight_key, bias_key = random.split(keys[i])
        
        if i == 0:  # First layer - first_layer_sine_init equivalent
            # First layer weights: uniform over [-1/n_in, 1/n_in]
            weights = jax.random.uniform(
                weight_key, 
                shape=(n_in, n_out),
                minval=-1/n_in, 
                maxval=1/n_in
            )
            # First layer biases: zeros
            biases = jnp.zeros((n_out,))
            
        else:  # Hidden layers - sine_init equivalent with fixed factor OMEGA_0
            # Hidden layer weights: uniform over [-sqrt(6/n_in)/OMEGA_0, sqrt(6/n_in)/OMEGA_0]
            bound = jnp.sqrt(6/n_in) / CONFIG.network.omega_0
            weights = jax.random.uniform(
                weight_key, 
                shape=(n_in, n_out),
                minval=-bound, 
                maxval=bound
            )
            # Hidden layer biases: zeros
            biases = jnp.zeros((n_out,))
        
        params.append([weights, biases])
    
    return params


def SIREN_neural_one_sample(x_input, params):
    """
    Forward pass for a single sample through SIREN network.
    
    Args:
        x_input: Input vector
        params: List of [weight, bias] pairs
    
    Returns:
        Output of the network
    """
    # First layer with sine activation
    x = jnp.sin(CONFIG.network.omega_0 * (x_input @ params[0][0] + params[0][1]))
    
    # Hidden layers with sine activation
    for i in range(1, len(params)-1):
        x = jnp.sin(CONFIG.network.omega_0 * (x @ params[i][0] + params[i][1]))
    
    # Output layer (no sine activation, no omega_0 scaling)
    output = (x @ params[-1][0]) + params[-1][1]
    
    return output


# Vectorized version for batch processing
SIREN_neural = jax.jit(vmap(SIREN_neural_one_sample, in_axes=(0, None)))