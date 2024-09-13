import chex
import haiku as hk
import jax

ForwardFn = hk.TransformedWithState
Model = tuple[hk.MutableParams, hk.MutableState]

Array = jax.Array
PRNGKey = chex.PRNGKey

Observation = Array

Value = Array
ValueVariance = Array
ExploitationPolicy = Array
ExplorationPolicy = Array
RewardVariance = Array
NetworkOutput = tuple[Value, ValueVariance, ExploitationPolicy, ExplorationPolicy, RewardVariance]
