import chex
import gym
import haiku as hk
import jax
import jax.numpy as jnp
import rlax
import mctx

from algorithms import nets
from algorithms import utils
from algorithms.types import ActorOutput, AgentOutput, Params, Tree



class GumbelAgent(object):
    """A MCTS agent."""

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Discrete, num_bins: int,
                 channels: int, use_v2: bool, output_init_scale: float, discount_factor: float, num_simulations: int,
                 max_search_depth: int):
        self._observation_space = observation_space
        self._action_space = action_space
        self._discount_factor = discount_factor
        self._num_simulations = num_simulations
        self._max_search_depth = num_simulations if max_search_depth is None else max_search_depth
        self.value_transform = utils.value_transform
        self.inv_value_transform = utils.inv_value_transform
        self._encode_fn = hk.without_apply_rng(hk.transform(
            lambda observations: nets.EZStateEncoder(channels, use_v2)(observations)))
        num_actions = self._action_space.n
        self._predict_fn = hk.without_apply_rng(hk.transform(
            lambda states: nets.EZPrediction(num_actions, num_bins, output_init_scale, use_v2)(states)))
        self._transit_fn = hk.without_apply_rng(hk.transform(
            lambda action, state: nets.EZTransition(use_v2)(action, state)))

    def init(self, rng_key: chex.PRNGKey):
        encoder_key, prediction_key, transition_key = jax.random.split(
            rng_key, 3)
        dummy_observation = self._observation_space.sample()
        encoder_params = self._encode_fn.init(encoder_key, dummy_observation)
        dummy_state = self._encode_fn.apply(encoder_params, dummy_observation)
        prediction_params = self._predict_fn.init(prediction_key, dummy_state)
        dummy_action = jnp.zeros((self._action_space.n,))
        transition_params = self._transit_fn.init(
            transition_key, dummy_action, dummy_state)
        params = Params(encoder=encoder_params,
                        prediction=prediction_params, transition=transition_params)
        return params

    def batch_step(self, rng_key: chex.PRNGKey, params: Params, timesteps: ActorOutput, temperature: float,
                   is_eval: bool):
        batch_size = timesteps.reward.shape[0]
        rng_key, step_key = jax.random.split(rng_key)
        step_keys = jax.random.split(step_key, batch_size)
        batch_root_step = jax.vmap(self._root_step, (0, None, 0, None, None))
        actions, agent_out = batch_root_step(
            step_keys, params, timesteps, temperature, is_eval)
        return rng_key, actions, agent_out

    def _root_step(self, rng_key: chex.PRNGKey, params: Params, timesteps: ActorOutput, temperature: float,
                   is_eval: bool):
        """The input `timesteps` is assumed to be [input_dim]."""
        trajectories = jax.tree_map(
            lambda t: t[None], timesteps)  # Add a dummy time dimension.
        agent_out = self.root_unroll(params, trajectories)
        # We do not need to squeeze the dummy time dimension
        policy_output = self.mcts_gumbel(rng_key, params, agent_out, is_eval)    
        policy_output = jax.tree_map(lambda t: t[0], policy_output)

        act_prob = policy_output.action_weights
        action = policy_output.action
        return action, agent_out

    def root_unroll(self, params: Params, trajectory: ActorOutput):
        """The input `trajectory` is assumed to be [T, input_dim].
            Is this guy batched already? Most probably YES.
        """
        state = self._encode_fn.apply(
            params.encoder, trajectory.observation)  # [T, S]
        logits, reward_logits, value_logits = self._predict_fn.apply(
            params.prediction, state)
        reward = utils.logits_to_scalar(reward_logits)
        reward = self.inv_value_transform(reward)
        value = utils.logits_to_scalar(value_logits)
        value = self.inv_value_transform(value)
        return AgentOutput(
            state=state,
            logits=logits,
            reward_logits=reward_logits,
            reward=reward,
            value_logits=value_logits,
            value=value,
        )

    def model_step(self, params: Params, state: chex.Array, action: chex.Array):
        """The input `state` and `action` are assumed to be [S] and []."""
        one_hot_action = hk.one_hot(action, self._action_space.n)
        next_state = self._transit_fn.apply(
            params.transition, one_hot_action, state)
        next_state = utils.scale_gradient(next_state, 0.5)
        logits, reward_logits, value_logits = self._predict_fn.apply(
            params.prediction, next_state)
        reward = utils.logits_to_scalar(reward_logits)
        reward = self.inv_value_transform(reward)
        value = utils.logits_to_scalar(value_logits)
        value = self.inv_value_transform(value)
        return AgentOutput(
            state=next_state,
            logits=logits,
            reward_logits=reward_logits,
            reward=reward,
            value_logits=value_logits,
            value=value,
        )

    def model_unroll(self, params: Params, state: chex.Array, action_sequence: chex.Array):
        """The input `state` and `action` are assumed to be [S] and [T]."""

        def fn(state: chex.Array, action: chex.Array):
            one_hot_action = hk.one_hot(action, self._action_space.n)
            next_state = self._transit_fn.apply(
                params.transition, one_hot_action, state)
            next_state = utils.scale_gradient(next_state, 0.5)
            return next_state, next_state

        _, state_sequence = jax.lax.scan(fn, state, action_sequence)
        logits, reward_logits, value_logits = self._predict_fn.apply(
            params.prediction, state_sequence)
        reward = utils.logits_to_scalar(reward_logits)
        reward = self.inv_value_transform(reward)
        value = utils.logits_to_scalar(value_logits)
        value = self.inv_value_transform(value)
        return AgentOutput(
            state=state_sequence,
            logits=logits,
            reward_logits=reward_logits,
            reward=reward,
            value_logits=value_logits,
            value=value,
        )

    def mcts_gumbel(self, rng_key: chex.PRNGKey, params: Params, root: AgentOutput, is_eval: bool):
        '''
            MCTS with Gumbel using MCTX library
            Input: 
                rng_key
                params : Params
                root: AgentOutput: logits [B x num_action], value [B], state [B x S]
            Returns:
                policy_output = { action, action_weights, search_tree }
        '''
        root_fn_out = mctx.RootFnOutput(
            prior_logits=   root.logits,
            value=          root.value,
            embedding=      root.state,
        )

        def recurrent_fn(params: Params, rng_key: chex.Array, action: chex.Array, embedding):
            '''
                One-step transition in the model for MCTS

                Prototype: function self.model_step()

                Returns:
                    mctx.RecurrentFnOutput = {reward, discount, prior_logits, value}
                    embedding: next state
            '''
            del rng_key
            res = jax.vmap(self.model_step, (None, 0, 0))(params, embedding, action)

            recurrent_fn_output = mctx.RecurrentFnOutput(
                reward=         res.reward,
                discount=       jnp.ones_like(res.reward) * self._discount_factor,
                prior_logits=   res.logits,
                value=          res.value,
            )
            return recurrent_fn_output, res.state
        
        # During evaluation choose action non-randomly but greedy
        if is_eval:
            gumbel_scale = 0.0
        else:
            gumbel_scale = 1.0

        policy_output = mctx.gumbel_muzero_policy(
            params=             params,
            rng_key=            rng_key,
            root=               root_fn_out,
            recurrent_fn=       recurrent_fn,
            num_simulations=    self._num_simulations,
            max_depth=          self._max_search_depth,
            gumbel_scale=       gumbel_scale
        )

        return policy_output