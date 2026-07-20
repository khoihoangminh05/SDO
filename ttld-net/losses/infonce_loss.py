"""InfoNCE contrastive loss for topology embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Topology contrastive objective with temperature tau=0.07."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.tau = temperature

    def forward(
        self,
        zi: torch.Tensor,
        z_pos: torch.Tensor,
        z_neg_list: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            zi: (B, N, D) anchor embeddings.
            z_pos: (B, N, D) positive context embeddings.
            z_neg_list: (B, N, M, D) negative contexts.
        """
        zi_norm = F.normalize(zi, dim=-1, eps=1e-6)
        zp_norm = F.normalize(z_pos, dim=-1, eps=1e-6)
        zn_norm = F.normalize(z_neg_list, dim=-1, eps=1e-6)

        pos_sim = (zi_norm * zp_norm).sum(-1) / self.tau
        neg_sim = torch.einsum("bnd,bnmd->bnm", zi_norm, zn_norm) / self.tau

        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        labels = torch.zeros(logits.shape[:2], dtype=torch.long, device=logits.device)
        return F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
        )
