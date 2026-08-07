#!/usr/bin/env python3
"""HD2 R2 loss/objective family (rev7). compute_loss(lg, tg, rH, w1, spec) -> scalar
for one H's reached subset. lg=logit, tg=up label {0,1}, rH=signed return,
w1=precomputed R1 |move| weight (fit-set p99 clip). Magnitude-awareness is the
user's idea ('bigger favorable move -> bigger reward, bigger adverse -> bigger
penalty'); realized via clip/rank/asym, NOT raw MSE (HM2: MSE-on-return is
magnitude-dominated)."""
from __future__ import annotations


def compute_loss(lg, tg, rH, w1, spec):
    import torch
    import torch.nn.functional as F
    name = (spec or {}).get("name", "R1")
    bce = F.binary_cross_entropy_with_logits

    if name == "R0":                       # plain BCE (control)
        return bce(lg, tg)
    if name == "R1":                       # BCE x |move| (fit-set clip) -- current
        return bce(lg, tg, weight=w1)
    if name == "R2_econ":                  # weight only the >=cost part of the move
        c = spec.get("cost", 0.0013)
        w = (rH.abs() - c).clamp(min=0.0)
        return bce(lg, tg, weight=w)
    if name == "focal":                    # down-weight easy examples
        g = spec.get("gamma", 2.0)
        p = torch.sigmoid(lg)
        pt = p * tg + (1 - p) * (1 - tg)
        ce = bce(lg, tg, reduction="none")
        return ((1 - pt).clamp(min=1e-6) ** g * ce).mean()
    if name == "asym":                     # wrong-DOWN penalized more (risk-averse, user)
        lam = spec.get("lam_down", 1.5)
        w = w1 * torch.where(rH < 0, torch.as_tensor(lam, device=lg.device),
                             torch.as_tensor(1.0, device=lg.device))
        return bce(lg, tg, weight=w)
    if name == "label_smooth":
        eps = spec.get("eps", 0.05)
        return bce(lg, tg * (1 - eps) + 0.5 * eps)
    if name == "IC":                       # maximize corr(pred, signed return) = -corr
        if lg.numel() < 2:
            return bce(lg, tg, weight=w1)  # fallback on tiny batch
        x = torch.sigmoid(lg)
        x = x - x.mean()
        y = rH - rH.mean()
        denom = (x.norm() * y.norm()).clamp_min(1e-8)
        return -(x * y).sum() / denom
    raise ValueError(f"unknown loss {name!r}")


# rev7 frozen loss variants (name -> spec)
VARIANTS = {
    "R0": {"name": "R0"},
    "R1": {"name": "R1"},                       # current default
    "R2_econ": {"name": "R2_econ", "cost": 0.0013},
    "focal": {"name": "focal", "gamma": 2.0},
    "asym": {"name": "asym", "lam_down": 1.5},  # user's down>up penalty
    "label_smooth": {"name": "label_smooth", "eps": 0.05},
    "IC": {"name": "IC"},                       # signed-return correlation
}
