import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
from utils.scalers import StandardScaler


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4 * h_dim),
            nn.GELU(),
            nn.Linear(4 * h_dim, h_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x)  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        ### transformer blocks
        #         input_seq_len = 3 * context_len
        input_seq_len = 2 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = True  # True for continuous actions

        ### prediction heads
        #         self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_out = torch.nn.Linear(h_dim, h_dim)

    #         self.predict_action = nn.Sequential(
    #             *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
    #         )

    def forward(self, timesteps, states, actions):
        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        #         returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

        #         stack rtg, states and actions and reshape sequence as
        #         (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        #         h = torch.stack(
        #             (returns_embeddings, state_embeddings, action_embeddings), dim=1
        #         ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)
        h = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 2 * T, self.h_dim)

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus
        # the 3 input variables at that timestep (r_t, s_t, a_t) in sequence.
        #         h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)
        h = h.reshape(B, T, 2, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        #         return_preds = self.predict_rtg(h[:,2])     # predict next rtg given r, s, a
        state_preds = self.predict_out(h[:, -1, -1])  # predict next state given r, s, a
        #         action_preds = self.predict_action(h[:,0])  # predict action given r, s

        return state_preds  # , action_preds#, return_preds


class TransformerDynamicsModel(nn.Module):
    def __init__(self, obs_shape, action_shape, n_blocks, h_dim, context_len, n_heads, lr, early_stop_patience, device):
        super().__init__()
        self.s_pred = nn.Sequential(nn.Linear(h_dim, 128), nn.ReLU(), nn.Linear(128, obs_shape)).cuda()
        self.r_pred = nn.Sequential(nn.Linear(h_dim, 128), nn.ReLU(), nn.Linear(128, 1)).cuda()

        self.dt = DecisionTransformer(
            state_dim=obs_shape,
            act_dim=action_shape,
            n_blocks=n_blocks,
            h_dim=h_dim,
            context_len=context_len,
            n_heads=n_heads,
            drop_p=0.1,
        )

        self.context_len = context_len
        self.scaler = StandardScaler()
        self.early_stop_patience = early_stop_patience

        self.to(device)

        self.params = list(self.dt.parameters()) + list(self.s_pred.parameters()) + list(self.r_pred.parameters())

        self.optimizer = torch.optim.Adam(
            self.params, lr=lr, weight_decay=1e-4
        )

        self.predict_difference = True
        self.reward_included = True

    def step(self, observations, actions):
        pass

    def train_single_step(self, replay_buffer, validation_ratio, batch_size):
        val_size = int(batch_size * validation_ratio)
        train_size = batch_size - val_size

        train_batch, val_batch = replay_buffer.random_split_transformer(
            val_size, batch_size * 10, self.context_len, self.scaler
        )

        train_input_s, train_input_a, train_target_s, train_target_r, train_ts = train_batch
        val_input_s, val_input_a, val_target_s, val_target_r, val_ts = val_batch

        train_size = train_input_s.shape[0]

        self.val_loss = 999
        loss_hist = []
        self.cnt = 0
        early_stop = False

        while not early_stop:
            for b in range(int(np.ceil(train_size / batch_size))):
                batch_idxs = np.random.choice(train_size, size=batch_size, replace=False)

                dt_out = self.dt(train_ts[batch_idxs], train_input_s[batch_idxs], train_input_a[batch_idxs])
                s_hats = self.s_pred(dt_out)
                r_hats = self.r_pred(dt_out)

                loss = F.mse_loss(s_hats, train_target_s[batch_idxs]) + F.mse_loss(r_hats, train_target_r[batch_idxs])

                self.optimizer.zero_grad()
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(self.params, 0.25)

                loss_hist.append(loss.item())
                self.optimizer.step()

            new_val_loss = self.evaluate(val_input_s, val_input_a, val_target_s, val_target_r, val_ts)
            early_stop = self._is_early_stop(new_val_loss)

        return loss_hist

    @torch.no_grad()
    def evaluate(self, val_input_s, val_input_a, val_target_s, val_target_r, val_ts):
        dt_out = self.dt(val_ts, val_input_s, val_input_a)
        s_hats = self.s_pred(dt_out)
        r_hats = self.r_pred(dt_out)

        loss = F.mse_loss(s_hats, val_target_s) + F.mse_loss(r_hats, val_target_r)
        return loss

    def _is_early_stop(self, new_val_loss):
        changed = False

        if (self.val_loss - new_val_loss) / self.val_loss > 0.01:
            changed = True
            self.val_loss = new_val_loss

        if changed:
            self.cnt = 0
        else:
            self.cnt += 1

        if self.cnt >= self.early_stop_patience:
            return True
        else:
            return False

    def preprocess_training_batch(self, data):
        # TODO: rename. also used for val
        """

        Args:
            data:
            predict_difference (bool): changes target. Should the dynamics models be trained to predict the **change**
                in the state?

        Returns:

        """
        # TODO: check if this still works well for traj data
        states, actions, next_states, rewards, not_dones = data

        inputs = torch.cat([states, actions], dim=-1)

        if self.predict_difference:
            target_state = next_states - states

        else:
            target_state = next_states

        if self.reward_included:
            target = torch.cat([target_state, rewards], dim=-1)

        else:
            target = target_state

        return inputs, target

    def step(self, states, actions):
        """
        Performs a single step

        Args:
            observations:
            actions:
            deterministic:

        Returns:

        """
        pass


    def imagine(self, rollout_batch_size, horizon, policy, env_replay_buffer, model_replay_buffer, termination_fn, rnd):
        # Starting the rollout from a real state, from anywhere in a trajectory
        # index [0] selects state from the given sample
        states = env_replay_buffer.sample(rollout_batch_size)[0]

        # This t will help us keep track of how much left zero-padding we need to do
        t = 1

        # These will be used to store previously seen states and chosen actions. Let's try and make this convenient
        # and only store the post-standardized data
        prev_state_buffer = []
        prev_action_buffer = []

        for i in range(horizon):
            with torch.no_grad():
                actions = policy(states).sample()

            if rnd:
                actions = td.Uniform(
                    low=torch.zeros_like(actions) - 1.,
                    high=torch.zeros_like(actions) + 1.
                ).sample().float().to(actions.device)

            # Scaling and then [B, z] -> [B, 1, z]
            sa = self.scaler.transform(torch.cat([states, actions], dim=-1))
            states = sa[:, :states.shape[-1]].unsqueeze(1)
            actions = sa[:, states.shape[-1]:].unsqueeze(1)

            # Each entry in these lists is a tensor of shape [B, 1, z]
            prev_state_buffer.append(states.clone())
            prev_action_buffer.append(actions.clone())

            if t < self.context_len:
                num_pads = self.context_len - t
                s_padding = torch.zeros(states.shape[0], num_pads, states.shape[-1]).to(states.device)
                a_padding = torch.zeros(actions.shape[0], num_pads, actions.shape[-1]).to(states.device)

                # [zero_pads, reversed version of buffer]
                states = torch.cat([s_padding, torch.cat(prev_state_buffer, dim=1)], dim=1)
                actions = torch.cat([a_padding, torch.cat(prev_action_buffer, dim=1)], dim=1)

            qqq

            next_states, rewards, dones, info = self.step(
                states, actions, deterministic=False
            )

            t += 1

            if termination_fn:
                dones = termination_fn(states, actions, next_states)

