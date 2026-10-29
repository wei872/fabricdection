"""Explicit schedule horizon and validation-based stopping; no model dependencies."""
import math


def cosine_factor(step, warmup_steps, decay_steps):
    if warmup_steps < 0 or decay_steps <= warmup_steps:
        raise ValueError('decay_steps must exceed nonnegative warmup_steps')
    if step < warmup_steps:
        return float(step + 1) / warmup_steps
    progress = max(0., min(1., (step - warmup_steps) / (decay_steps - warmup_steps)))
    return .05 + .95 * .5 * (1. + math.cos(math.pi * progress))


class EarlyStopper:
    """Patience counts validation checks after min_epochs, not training loss."""
    def __init__(self, patience=0, min_delta=.001, min_epochs=20):
        if patience < 0 or min_epochs < 0 or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError('early-stop patience/min_epochs/min_delta must be finite and nonnegative')
        self.patience, self.min_delta, self.min_epochs = int(patience), float(min_delta), int(min_epochs)
        self.best, self.bad_checks = None, 0

    def step(self, score, epoch):
        if not math.isfinite(score):
            raise ValueError('early-stop validation score must be finite')
        if self.best is None or score > self.best + self.min_delta:
            self.best, self.bad_checks = float(score), 0
        elif epoch >= self.min_epochs:
            self.bad_checks += 1
        else:
            self.bad_checks = 0
        return bool(self.patience and epoch >= self.min_epochs and self.bad_checks >= self.patience)

    def state_dict(self):
        return dict(patience=self.patience, min_delta=self.min_delta, min_epochs=self.min_epochs,
                    best=self.best, bad_checks=self.bad_checks)

    def load_state_dict(self, state):
        self.best = state.get('best')
        self.bad_checks = int(state.get('bad_checks', 0))


def schedule_spec(epochs, n_batches, warmup_epochs, cfg_decay=None, explicit_decay=None, saved=None):
    """Resume preserves the ORIGINAL step horizon even when max epochs changes."""
    if epochs < 1 or n_batches < 1:
        raise ValueError('epochs and n_batches must be positive')
    if explicit_decay is not None and explicit_decay < 1:
        raise ValueError('lr-decay-epochs must be positive')
    if saved is not None:
        existing = saved.get('lr_schedule')
        if existing is not None:
            result = dict(existing)
        else:
            # Backward compatibility for previous LambdaLR / CosineAnnealingLR.
            batches = int(saved.get('batches_per_epoch', n_batches))
            old_epochs = int(saved.get('epochs', epochs))
            warmup = int(float(warmup_epochs) * batches)
            result = dict(kind='warmup_cosine_floor' if warmup else 'legacy_cosine_annealing',
                          decay_epochs=old_epochs, warmup_steps=warmup,
                          decay_steps=max(warmup + 1, old_epochs * batches))
        if explicit_decay is not None and explicit_decay != result['decay_epochs']:
            raise ValueError('Cannot change LR horizon during --resume; use --pretrained with a new --out')
        return result
    horizon = explicit_decay if explicit_decay is not None else (cfg_decay if cfg_decay is not None else epochs)
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError('lr_decay_epochs must be a positive integer')
    horizon = min(epochs, horizon)
    warmup = int(float(warmup_epochs) * n_batches)
    return dict(kind='warmup_cosine_floor', decay_epochs=horizon, warmup_steps=warmup,
                decay_steps=max(warmup + 1, horizon * n_batches))
