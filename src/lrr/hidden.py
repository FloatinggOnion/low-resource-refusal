"""H readings: hidden states at the last token of the chat-templated prompt.

Position choice (recorded, fixed in advance): the final token of the string produced by
apply_chat_template(..., add_generation_prompt=True, enable_thinking=False). For Qwen3 that is the
newline after the empty <think></think> block, i.e. the position from which the first response
token is predicted. All layers are kept: index 0 = embedding output, index L = output of the
final norm (HF convention).
"""

from __future__ import annotations

import numpy as np
import torch

from .probe import chat_text, encode_batch, make_batches


@torch.no_grad()
def last_token_states(model, tok, user_texts: list[str], max_tokens: int = 8192, max_bs: int = 32) -> np.ndarray:
    """(n, L+1, d) float32."""
    texts = [chat_text(tok, u) for u in user_texts]
    lens = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
    out = None
    for idx in make_batches(lens, max_tokens, max_bs):
        input_ids, attn, pos = encode_batch(tok, [texts[i] for i in idx], model.device)
        res = model(input_ids=input_ids, attention_mask=attn, position_ids=pos, output_hidden_states=True, logits_to_keep=1)
        hs = torch.stack([h[:, -1, :] for h in res.hidden_states], dim=1).float().cpu().numpy()  # (b, L+1, d)
        if out is None:
            out = np.zeros((len(texts),) + hs.shape[1:], dtype=np.float32)
        out[idx] = hs
    return out


def token_counts(tok, user_texts: list[str]) -> dict[str, np.ndarray]:
    """Content tokens (the raw prompt only) and total templated tokens, per prompt."""
    content = np.array([len(tok(u, add_special_tokens=False).input_ids) for u in user_texts])
    total = np.array([len(tok(chat_text(tok, u), add_special_tokens=False).input_ids) for u in user_texts])
    return {"content": content, "total": total}
