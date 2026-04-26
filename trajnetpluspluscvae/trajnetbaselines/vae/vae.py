import copy
import itertools

import numpy as np
import torch

import trajnetplusplustools

from .. import augmentation
from ..lstm.lstm import generate_pooling_inputs
from ..lstm.modules import Hidden2Normal, InputEmbedding
from ..lstm.utils import center_scene

NAN = float('nan')


def drop_distant(xy, r=6.0):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r ** 2
    return xy[:, mask], mask


class MLPEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs):
        return self.net(inputs)


class CVAE(torch.nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128, pool=None, pool_to_input=True, goal_dim=None, goal_flag=False,
                 num_modes=1, latent_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.num_modes = num_modes
        self.pool = pool
        self.pool_to_input = pool_to_input

        scale = 4.0
        self.input_embedding = InputEmbedding(2, self.embedding_dim, scale)

        self.goal_flag = goal_flag
        self.goal_dim = goal_dim or embedding_dim
        self.goal_embedding = InputEmbedding(2, self.goal_dim, scale)
        goal_rep_dim = self.goal_dim if self.goal_flag else 0

        self.pooling_dim = 0
        if pool is not None and self.pool_to_input:
            self.pooling_dim = self.pool.out_dim

        encoder_input_dim = self.embedding_dim + goal_rep_dim + self.pooling_dim
        decoder_input_dim = encoder_input_dim + self.latent_dim

        self.obs_encoder = torch.nn.LSTMCell(encoder_input_dim, self.hidden_dim)
        self.future_encoder = torch.nn.LSTMCell(encoder_input_dim, self.hidden_dim)
        self.decoder = torch.nn.LSTMCell(decoder_input_dim, self.hidden_dim)
        self.hidden2normal = Hidden2Normal(self.hidden_dim)

        self.prior_net = MLPEncoder(self.hidden_dim, self.hidden_dim, 2 * self.latent_dim)
        self.posterior_net = MLPEncoder(2 * self.hidden_dim, self.hidden_dim, 2 * self.latent_dim)

    def _init_hidden_cell_state(self, num_tracks, device):
        return (
            [torch.zeros(self.hidden_dim, device=device) for _ in range(num_tracks)],
            [torch.zeros(self.hidden_dim, device=device) for _ in range(num_tracks)],
        )

    def _masked_hidden_state(self, hidden_cell_state, track_mask):
        return [
            torch.stack([h for m, h in zip(track_mask, hidden_cell_state[0]) if m], dim=0),
            torch.stack([c for m, c in zip(track_mask, hidden_cell_state[1]) if m], dim=0),
        ]

    def _goal_embedding(self, obs2, goals, track_mask):
        norm_factors = torch.norm(obs2 - goals, dim=1)
        goal_direction = (obs2 - goals) / norm_factors.unsqueeze(1).clamp_min(1e-8)
        goal_direction[norm_factors == 0] = torch.tensor([0.0, 0.0], device=obs2.device)
        goal_direction = goal_direction[track_mask]
        return self.goal_embedding(goal_direction)

    def _decoder_latent_input(self, latent, batch_split, num_tracks, device):
        latent_per_track = torch.zeros(num_tracks, self.latent_dim, device=device)
        latent_per_track[batch_split[:-1]] = latent
        return latent_per_track

    def _distribution_parameters(self, encoded):
        mu, log_var = torch.chunk(encoded, 2, dim=1)
        log_var = torch.clamp(log_var, min=-6.0, max=4.0)
        return mu, log_var

    def _sample_latent(self, mu, log_var):
        if self.training:
            epsilon = torch.randn_like(mu)
            return mu + torch.exp(0.5 * log_var) * epsilon
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

    def _pool(self, obs1, obs2, hidden_cell_state, track_mask, batch_split):
        curr_positions, prev_positions, curr_hidden_state, track_mask_positions = generate_pooling_inputs(
            obs2, obs1, hidden_cell_state, track_mask, batch_split
        )
        pool_sample = self.pool(curr_hidden_state, prev_positions, curr_positions)
        return pool_sample[track_mask_positions.view(-1)]

    def step(self, lstm, hidden_cell_state, obs1, obs2, goals, batch_split, latent=None):
        track_mask = (torch.isnan(obs1[:, 0]) + torch.isnan(obs2[:, 0])) == 0
        hidden_cell_stacked = self._masked_hidden_state(hidden_cell_state, track_mask)

        curr_velocity = (obs2 - obs1)[track_mask]
        input_emb = self.input_embedding(curr_velocity)

        if self.goal_flag:
            goal_emb = self._goal_embedding(obs2, goals, track_mask)
            input_emb = torch.cat([input_emb, goal_emb], dim=1)

        if self.pool is not None:
            pooled = self._pool(obs1, obs2, hidden_cell_state, track_mask, batch_split)
            if self.pool_to_input:
                input_emb = torch.cat([input_emb, pooled], dim=1)
            else:
                hidden_cell_stacked[0] += pooled

        if latent is not None:
            input_emb = torch.cat([input_emb, latent[track_mask]], dim=1)

        hidden_cell_stacked = lstm(input_emb, hidden_cell_stacked)
        normal_masked = self.hidden2normal(hidden_cell_stacked[0])

        normal = torch.full((track_mask.size(0), 5), NAN, device=obs1.device)
        mask_index = [i for i, m in enumerate(track_mask) if m]
        for i, h, c, n in zip(mask_index, hidden_cell_stacked[0], hidden_cell_stacked[1], normal_masked):
            hidden_cell_state[0][i] = h
            hidden_cell_state[1][i] = c
            normal[i] = n

        return hidden_cell_state, normal

    def _encode_sequence(self, lstm, hidden_cell_state, sequence, goals, batch_split):
        for obs1, obs2 in zip(sequence[:-1], sequence[1:]):
            hidden_cell_state, _ = self.step(lstm, hidden_cell_state, obs1, obs2, goals, batch_split)
        return hidden_cell_state

    def _primary_context(self, hidden_cell_state, batch_split):
        primary_ids = batch_split[:-1]
        return torch.stack([hidden_cell_state[0][idx] for idx in primary_ids], dim=0)

    def _posterior_prior(self, obs_hidden, future_hidden=None):
        prior_mu, prior_log_var = self._distribution_parameters(self.prior_net(obs_hidden))

        posterior_mu, posterior_log_var = prior_mu, prior_log_var
        if future_hidden is not None:
            posterior_inputs = torch.cat([obs_hidden, future_hidden], dim=1)
            posterior_mu, posterior_log_var = self._distribution_parameters(self.posterior_net(posterior_inputs))

        return {
            'prior_mu': prior_mu,
            'prior_log_var': prior_log_var,
            'posterior_mu': posterior_mu,
            'posterior_log_var': posterior_log_var,
        }

    def _decode_modes(self, hidden_cell_state, obs_normals, positions_seed, prediction_truth, goals, batch_split, latent_params):
        num_tracks = len(hidden_cell_state[0])
        normals = {mode: list(obs_normals) for mode in range(self.num_modes)}
        positions = {mode: list(positions_seed) for mode in range(self.num_modes)}

        for mode in range(self.num_modes):
            if self.training:
                latent = self._sample_latent(latent_params['posterior_mu'], latent_params['posterior_log_var'])
            else:
                latent = self._sample_latent(latent_params['prior_mu'], latent_params['prior_log_var'])

            latent_per_track = self._decoder_latent_input(latent, batch_split, num_tracks, goals.device)
            hidden_cell_state_dec = (
                [h.clone() for h in hidden_cell_state[0]],
                [c.clone() for c in hidden_cell_state[1]],
            )

            for obs1, obs2 in zip(prediction_truth[:-1], prediction_truth[1:]):
                if obs1 is None:
                    obs1 = positions[mode][-2].detach()
                else:
                    obs1 = obs1.clone()
                    for primary_id in batch_split[:-1]:
                        obs1[primary_id] = positions[mode][-2][primary_id].detach()
                if obs2 is None:
                    obs2 = positions[mode][-1].detach()
                else:
                    obs2 = obs2.clone()
                    for primary_id in batch_split[:-1]:
                        obs2[primary_id] = positions[mode][-1][primary_id].detach()

                hidden_cell_state_dec, normal = self.step(
                    self.decoder, hidden_cell_state_dec, obs1, obs2, goals, batch_split, latent=latent_per_track
                )
                normals[mode].append(normal)
                positions[mode].append(obs2 + normal[:, :2])

        rel_pred_scene = [torch.stack(normals[mode], dim=0) for mode in normals]
        pred_scene = [torch.stack(positions[mode], dim=0) for mode in positions]
        return rel_pred_scene, pred_scene

    def forward(self, observed, goals, batch_split, prediction_truth=None, n_predict=None):
        assert ((prediction_truth is None) + (n_predict is None)) == 1
        if n_predict is not None:
            prediction_truth = [None for _ in range(n_predict - 1)]

        num_tracks = observed.size(1)
        hidden_cell_state = self._init_hidden_cell_state(num_tracks, observed.device)

        if self.pool is not None:
            max_num_neighbor = (batch_split[1:] - batch_split[:-1]).max() - 1
            batch_size = len(batch_split) - 1
            self.pool.reset(batch_size * (max_num_neighbor + 1), max_num_neighbor, device=observed.device)

        obs_normals = []
        positions_seed = []
        for obs1, obs2 in zip(observed[:-1], observed[1:]):
            hidden_cell_state, normal = self.step(self.obs_encoder, hidden_cell_state, obs1, obs2, goals, batch_split)
            obs_normals.append(normal)
            positions_seed.append(obs2 + normal[:, :2])

        prediction_truth = copy.deepcopy(list(itertools.chain.from_iterable((observed[-1:], prediction_truth))))

        future_hidden = None
        if self.training:
            future_hidden_cell_state = self._init_hidden_cell_state(num_tracks, observed.device)
            future_hidden_cell_state = self._encode_sequence(
                self.future_encoder, future_hidden_cell_state, prediction_truth, goals, batch_split
            )
            future_hidden = self._primary_context(future_hidden_cell_state, batch_split)

        obs_hidden = self._primary_context(hidden_cell_state, batch_split)
        latent_params = self._posterior_prior(obs_hidden, future_hidden)
        rel_pred_scene, pred_scene = self._decode_modes(
            hidden_cell_state, obs_normals, positions_seed, prediction_truth, goals, batch_split, latent_params
        )

        return rel_pred_scene, pred_scene, latent_params


class VAEPredictor(object):
    def __init__(self, model):
        self.model = model

    def save(self, state, filename):
        with open(filename, 'wb') as f:
            torch.save(self, f)

        with open(filename + '.state', 'wb') as f:
            torch.save(state, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return torch.load(f, weights_only=False)

    def __call__(self, paths, scene_goal, n_predict=12, modes=1, predict_all=True, obs_length=9, start_length=0, args=None):
        del predict_all
        self.model.eval()
        self.model.num_modes = modes
        with torch.no_grad():
            xy = trajnetplusplustools.Reader.paths_to_xy(paths)
            batch_split = [0, xy.shape[1]]

            if args.normalize_scene:
                xy, rotation, center, scene_goal = center_scene(xy, obs_length, goals=scene_goal)

            xy = torch.Tensor(xy)
            scene_goal = torch.Tensor(scene_goal)
            batch_split = torch.Tensor(batch_split).long()

            multimodal_outputs = {}
            _, output_scenes_list, _ = self.model(xy[start_length:obs_length], scene_goal, batch_split, n_predict=n_predict)
            for mode_idx, output_scenes in enumerate(output_scenes_list):
                output_scenes = output_scenes.numpy()
                if args.normalize_scene:
                    output_scenes = augmentation.inverse_scene(output_scenes, rotation, center)
                output_primary = output_scenes[-n_predict:, 0]
                output_neighs = output_scenes[-n_predict:, 1:]
                multimodal_outputs[mode_idx] = [output_primary, output_neighs]

        return multimodal_outputs


VAE = CVAE
