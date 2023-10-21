# Copyright (c) Facebook, Inc. and its affiliates.
# Modified from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
# from torch.cuda.amp import autocast
# from detectron2.projects.point_rend.point_features import point_sample


def soft_iou(mask1: torch.Tensor, mask2: torch.Tensor):
    mask1, mask2 = mask1.flatten(1), mask2.flatten(1)
    intersection = torch.einsum("nc,mc->nm", mask1, mask2)
    union = mask1.sum(-1)[:, None] + mask2.sum(-1)[None, :] - intersection
    siou = intersection / (union + 1e-8)
    return siou


soft_iou_jit = torch.jit.script(
    soft_iou
)


def pair_soft_iou(mask1: torch.Tensor, mask2: torch.Tensor):
    mask1, mask2 = mask1.flatten(1), mask2.flatten(1)
    intersection = (mask1 * mask2).sum(-1)
    union = mask1.sum(-1) + mask2.sum(-1) - intersection
    siou = (intersection / union)
    return siou


pair_soft_iou_jit = torch.jit.script(
    pair_soft_iou
)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cost_class:float=1, cost_mask:float=1, num_points:int=0):
        """Creates the matcher
        Args:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask

        assert cost_class != 0 or cost_mask != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def forward(self, out_cap, out_mask, tgt_cap, tgt_mask):
        """Performs memory-friendly matching
        Args:
            out_cap: [num_queries, embed_dim]
            out_mask: [num_queries, H_pred, W_pred]
            tar_cap: [num_target_queries, embed_dim]
            tgt_mask: [num_target_queries, H_gt, W_gt]
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j)
            where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_queries)
        """
        assert out_cap.shape[0] == out_mask.shape[0], \
            "out_cap and out_mask must have same number of queries. but got {} and {}".format(out_cap.shape[0], out_mask.shape[0])
        assert tgt_cap.shape[0] == tgt_mask.shape[0], \
            "tgt_cap and tgt_mask must have same number of queries. but got {} and {}".format(tgt_cap.shape[0], tgt_mask.shape[0])
        num_queries = out_cap.shape[0]

        # sim = torch.einsum(
        #     "nc,mc->nm",
        #     out_cap / out_cap.norm(dim=-1, keepdim=True),
        #     tgt_cap / tgt_cap.norm(dim=-1, keepdim=True)
        # )
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # cost_class = sim

        cost_mask = soft_iou_jit(out_mask, tgt_mask)

        # cost matrix
        C = self.cost_mask * cost_mask
        C = C.reshape(num_queries, -1).cpu()
        i, j = linear_sum_assignment(C, maximize=True)

        # filter matching with soft iou
        # TODO: change to a better threshold
        valid_mask = (pair_soft_iou_jit(out_mask[i], tgt_mask[j]) > 0.1).cpu()
        return (
            torch.as_tensor(i, dtype=torch.int64)[valid_mask],
            torch.as_tensor(j, dtype=torch.int64)[valid_mask]
        )
