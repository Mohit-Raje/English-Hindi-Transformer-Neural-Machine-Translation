import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gradio as gr
import sentencepiece as spm
from transformer import Transformer  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2


def create_padding_mask(seq, pad=PAD_ID):
    return (seq != pad).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    mask = torch.tril(torch.ones((size, size), dtype=torch.uint8))
    return mask.unsqueeze(0).unsqueeze(1) == 1


# Load Tokenizers
sp_en = spm.SentencePieceProcessor(model_file="tokenizers/spm_en.model")
sp_hi = spm.SentencePieceProcessor(model_file="tokenizers/spm_hi.model")


# Load Transformer Model
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048

model = Transformer(
    src_vocab_size=len(sp_en),
    tgt_vocab_size=len(sp_hi),
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff
).to(device)

checkpoint = torch.load("model/transformer_en_hi.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# Translation Function
def translate_sentence(sentence):
    sentence = sentence.strip()
    if sentence == "":
        return "Please enter text."

    # Encode English sentence
    src_ids = [SOS_ID] + sp_en.encode(sentence, out_type=int) + [EOS_ID]
    src = torch.tensor(src_ids).unsqueeze(0).to(device)

    # Encoder mask
    src_mask = create_padding_mask(src)

    # Encoder output
    enc_output = model.encoder(src, src_mask)

    # Decoder input starts with <SOS>
    tgt = torch.tensor([[SOS_ID]], dtype=torch.long).to(device)

    # Autoregressive decoding (greedy)
    for _ in range(50):  # max length
        # look-ahead mask
        tgt_mask = create_look_ahead_mask(tgt.size(1)).to(device)

        # Decoder forward
        dec_output = model.decoder(tgt, enc_output, tgt_mask, src_mask)

        # Project to vocab
        logits = model.fc_out(dec_output)

        # Pick next token
        next_id = torch.argmax(logits[:, -1, :], dim=-1).item()

        if next_id == EOS_ID:
            break

        # append
        tgt = torch.cat(
            [tgt, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1
        )

    # Convert IDs → Hindi text
    out_ids = [i for i in tgt.squeeze().tolist() if i not in [SOS_ID, EOS_ID, PAD_ID]]
    hindi = sp_hi.decode(out_ids)
    return hindi

# Gradio Web Interface
demo = gr.Interface(
    fn=translate_sentence,
    inputs=gr.Textbox(lines=3, label="English Input"),
    outputs=gr.Textbox(lines=3, label="Hindi Translation"),
    title="English → Hindi Neural Machine Translation",
    description="A fully custom Transformer model built and trained from scratch in PyTorch, featuring 564M trainable parameters and complete low-level implementation of attention and encoder–decoder architecture.",
    examples=[
        ["I love you"],
        ["Broken symbolic link"],
        ["What is your name?"],
        ["JavaScript Support Plugin"],
    ]
)

demo.launch()
