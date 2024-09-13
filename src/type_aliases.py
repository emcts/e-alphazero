import haiku as hk
import jax
import chex

ForwardFn = hk.TransformedWithState
Model = tuple[hk.MutableParams, hk.MutableState]

Array = chex.Array
PRNGKey = chex.PRNGKey
