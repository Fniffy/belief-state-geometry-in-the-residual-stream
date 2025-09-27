import torch
import torch.nn as nn
import torch.optim as optim
from parentheses import *
from constants import *
from transformer_lens import HookedTransformer, HookedTransformerConfig


# toy vocab
stoi = {"(":0, ")":1, "?":2, "<pad>":3}
itos = {v:k for k,v in stoi.items()}

vocab = {'(':0, ')':1, '[':2, ']':3, '{':4, '}':5, '?':6}
inv_vocab = {v:k for k,v in vocab.items()}

def tokenize(seq):
    return [vocab[c] for c in seq]

def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def make_batch(batch_size = BATCH_SIZE):
    xs, ys = [], []
    for _ in range(batch_size):
        s, target = generate_testcase_with_deletion(1)
        xs.append(encode(s))
        ys.append(stoi[target])
    # gleiche Länge durch Padding
    max_len = max(len(x) for x in xs)
    xs = [torch.cat([x, torch.full((max_len - len(x),), stoi["<pad>"])]) for x in xs]
    return torch.stack(xs), torch.tensor(ys)

if __name__ =="__main__":
    

    model_cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=64,
        n_heads=1,
        d_head=64,
        n_ctx=32,
        d_vocab=len(stoi),
        act_fn="relu",
    )

    model = HookedTransformer(model_cfg)

    # 3. Training loop
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(1000):
        x, y = make_batch()  # Shape: [batch_size, seq_len]
        logits = model(x)
        idxs = (x == stoi["?"]).nonzero(as_tuple=True)
        # logits an den "?"-Positionen extrahieren
        preds = logits[idxs[0], idxs[1]]  # Shape: [batch_size, vocab]
        loss = loss_fn(preds, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(step, loss.item())
