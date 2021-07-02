import torch
from tqdm.notebook import tqdm
import numpy as np

def flatten(l):
    return [item for sublist in l for item in sublist]

def remove_tech_tokens(mystr, tokens_to_remove=['<eos>', '<sos>', '<unk>', '<pad>']):
    return [x for x in mystr if x not in tokens_to_remove]


def get_text(x, TRG_vocab):
    text = [TRG_vocab.itos[token] for token in x]
    try:
        end_idx = text.index('<eos>')
        text = text[:end_idx]
    except ValueError:
        pass
    text = remove_tech_tokens(text)
    if len(text) < 1:
        text = []
    return text


def generate_translation(model, iterator, TRG_vocab):
    model.eval()
    original_text = []
    generated_text = []
    model.eval()
    with torch.no_grad():

        for i, batch in tqdm(enumerate(iterator)):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg sent len, batch size]
            #output = [trg sent len, batch size, output dim]

            output = output.argmax(dim=-1)
        
            original_text.extend([get_text(x, TRG_vocab) for x in trg.cpu().numpy().T])
            generated_text.extend([get_text(x, TRG_vocab) for x in output[1:].detach().cpu().numpy().T])
    return (original_text, generated_text)        


def emb_sentence(sent, dicts):
    emb = dicts[0]
    no_emb = dicts[1]
    vocab = dicts[2]
    result = [(emb[(vocab.itos[idx])]) if (vocab.itos[idx] in emb.vocab.keys()) else no_emb[vocab.itos[idx]] for idx in sent]
    #print(result[0])
    return np.array(result)


def generate_translation_2(model, iterator, TRG_vocab, embedding_pretrained = False, dicts = None):
    model.eval()
    original_text = []
    generated_text = []
    model.eval()
    with torch.no_grad():

        for i, batch in tqdm.tqdm(enumerate(iterator)):

            src = batch.src
            trg = batch.trg

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if embedding_pretrained == True:
                sentences = src.cpu().numpy()[:]
                src = [emb_sentence(x, dicts) for x in sentences]
                src = torch.from_numpy(np.array(src)).to(device)
                src = src.float()

            output, _ = model(src, trg[:, :-1])
            trg_pred = output.argmax(dim=2).cpu().numpy()
            orig =  trg.cpu().numpy()
            for orig_txt, gen_txt in zip(orig, trg_pred):
                original_text.append(get_text(orig_txt, TRG_vocab))
                generated_text.append(get_text(gen_txt, TRG_vocab))

    return (original_text, generated_text)        


