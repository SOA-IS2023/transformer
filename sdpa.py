import torch
import torch.nn as nn


def sdpa(queries: torch.tensor, keys: torch.tensor, values: torch.tensor):
    dk = queries.size(dim=1)

    qk_mult = torch.dot(queries, torch.t(keys))*(1/torch.sqrt(dk))
    softmax = nn.Softmax(dim=1)
    result = torch.dot(softmax(qk_mult), values)

    return result

def multi_head(queries: torch.tensor, keys: torch.tensor, values: torch.tensor, num_heads: int):
    d_model = queries.size(dim=1)
    assert d_model % num_heads == 0
    depth = d_model // num_heads




