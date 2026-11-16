"""LeRobot's async policy server, fixed for Diffusion Policy checkpoints.

Run it exactly like the stock server, from the lerobot venv:

    ~/abhay_ws/lerobot/.venv/bin/python scripts/diffusion_policy_server.py \
        --host=127.0.0.1 --port=8080 --fps=15

Why: the stock server calls `policy.predict_action_chunk(obs)` directly with one observation and
never fills the policy's history queues. Diffusion's offline path then adds the time axis to the
images but NOT to `observation.state`, so a (1, 7) state is read as batch 1 x 7 observation steps
and `generate_actions` fails its `n_obs_steps` assertion. This adds the missing axis.

Only valid for checkpoints trained with n_obs_steps=1 (scripts/train_diffusion_local.sh): the
client sends one observation per chunk, so there is no real history to give a 2-step model.
Every other policy type is untouched.
"""

from lerobot.async_inference.policy_server import serve
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import OBS_STATE

_orig = DiffusionPolicy.predict_action_chunk


def _predict_action_chunk(self, batch, noise=None):
    queues_empty = not any(len(q) > 0 for q in self._queues.values())
    if queues_empty and OBS_STATE in batch and batch[OBS_STATE].ndim == 2:
        if self.config.n_obs_steps != 1:
            raise RuntimeError(
                f'diffusion checkpoint has n_obs_steps={self.config.n_obs_steps}; this server sends one '
                'observation per chunk, so only n_obs_steps=1 checkpoints can be served')
        batch = dict(batch)
        batch[OBS_STATE] = batch[OBS_STATE].unsqueeze(1)
    return _orig(self, batch, noise=noise)


DiffusionPolicy.predict_action_chunk = _predict_action_chunk

if __name__ == '__main__':
    serve()
