import torch

def tokenize(fasta_path):

    with open(fasta_path, "r") as F:
        all_proteins = F.read()

    vocab = sorted(set(all_proteins))

    stoi = {c:i for i, c in enumerate(vocab)}
    itos = {i:c for i, c in enumerate(vocab)}
    encoder = lambda s: [stoi[x] for x in s]
    decoder = lambda s: "".join([itos[x] for x in s])

    data = torch.tensor(encoder(all_proteins), dtype = torch.long)

    return data, encoder, decoder

def get_batch(data, block_size, batch_size):
    
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.vstack([data[i:i+block_size] for i in ix])
    y = torch.vstack([data[i+1:i+block_size+1] for i in ix])

    return x, y