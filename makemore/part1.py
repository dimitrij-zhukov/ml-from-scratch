import torch

words = open('names.txt', 'r').read().splitlines()

N = torch.zeros((27, 27), dtype = torch.int32)

chars = sorted(list(set(''.join(words))))

stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0

itos = {i:s for s, i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        N[ix1, ix2] += 1

N = N + 1

P = N.float() / N.float().sum(1, keepdim=True)

log_likelihood = 0.0
n = 0
for w in ["anrejq"]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        p = P[ix1, ix2]
        logprob = torch.log(p)
        log_likelihood += logprob
        n += 1


nll = -log_likelihood
norm = nll / n

print(norm)

g = torch.Generator().manual_seed(2147483647)

ws = []
for _ in range(10000):
    ix = 0
    out = []
    while True:
        p = P[ix]

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])

        if ix == 0:
            break
    l = ''.join(out)
    ws.append(l)

amount = 0

for w in ws:
    for ch in w:
        amount += 1
    amount -= 1

print(amount / len(ws))