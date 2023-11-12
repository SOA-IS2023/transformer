import torch
import torch.nn as nn


def sdpa(queries: torch.tensor, keys: torch.tensor, values: torch.tensor):
    dk = queries.size(dim=1)

    qk_mult = torch.dot(queries, torch.t(keys))*(1/torch.sqrt(dk))
    softmax = nn.Softmax(dim=1)
    result = torch.dot(softmax(qk_mult), values)

    return result

def multi_head_attention(query, key, value, num_heads):
    d_model = query.shape[-1]
    assert d_model % num_heads == 0
    depth = d_model // num_heads

    # Linearly project the inputs for each head
    query = np.reshape(np.dot(query, np.random.rand(d_model, d_model)), (query.shape[0], num_heads, -1, depth))
    key = np.reshape(np.dot(key, np.random.rand(d_model, d_model)), (key.shape[0], num_heads, -1, depth))
    value = np.reshape(np.dot(value, np.random.rand(d_model, d_model)), (value.shape[0], num_heads, -1, depth))

    # Apply scaled dot-product attention for each head
    attention_outputs = []
    attention_weights = []
    for i in range(num_heads):
        output, weights = scaled_dot_product_attention(query[:, i], key[:, i], value[:, i])
        attention_outputs.append(output)
        attention_weights.append(weights)

    # Concatenate the attention outputs from all heads
    attention_outputs = np.concatenate(attention_outputs, axis=-2)

    # Linearly project the concatenated outputs
    output = np.dot(attention_outputs, np.random.rand(d_model, d_model))

    return output, attention_weights