import torch
import torch.nn.functional as F


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def infer(input, model, tokenizer, device=get_device()):
    prepared_sents = []
    for inp in input:
        inp = tokenizer.tokenize(inp)
        if len(inp) > tokenizer.max_len:
            inp = inp[:tokenizer.max_len]
        else:
            inp = F.pad(
                inp,
                (0, tokenizer.max_len-len(inp)),
                mode='constant',
                value=0,
            )
        prepared_sents.append(inp)
    input_ids = torch.stack(prepared_sents).to(device)
    with torch.no_grad():
        _, probs = model(input_ids)
    preds = tokenizer.detokenize(
        torch.argmax(probs, dim=1).cpu().tolist(),
    )
    return preds
