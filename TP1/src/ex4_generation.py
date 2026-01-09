import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

SEED = 42
torch.manual_seed(SEED)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=50,
)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)


def generate_once(seed):
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

for s in [1, 2, 3, 4, 5]:
    print("SEED", s)
    print(generate_once(s))
    print("-" * 40)

def generate_sampling(seed, repetition_penalty=None):
    torch.manual_seed(seed)
    gen_kwargs = dict(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    out = model.generate(**gen_kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True)

seed = 3  # choisis un seed où tu voyais des répétitions (ex: 3 ou 4)

print("=== Sans pénalité ===")
print(generate_sampling(seed, repetition_penalty=None))
print("\n=== Avec repetition_penalty=2.0 ===")
print(generate_sampling(seed, repetition_penalty=2.0))

def generate_temp(seed, temperature):
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

seed = 2  # tu peux garder 2 (ou un autre), mais garde le même pour comparer

print("=== Temperature = 0.1 ===")
print(generate_temp(seed, 0.1))
print("\n=== Temperature = 2.0 ===")
print(generate_temp(seed, 2.0))


out_beam = model.generate(
    **inputs,
    max_length=50,
    num_beams=5,
    early_stopping=True
)
txt_beam = tokenizer.decode(out_beam[0], skip_special_tokens=True)
print("\n=== Beam search (num_beams=5) ===")
print(txt_beam)




def beam_generate(num_beams):
    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_length=50,
        num_beams=num_beams,
        early_stopping=True
    )
    dt = time.perf_counter() - t0
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    return dt, txt

for b in [5, 10, 20]:
    dt, txt = beam_generate(b)
    print(f"\n=== Beam search num_beams={b} ===")
    print(f"Time: {dt:.3f} s")
    print(txt)
