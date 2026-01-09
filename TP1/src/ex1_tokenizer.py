from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

phrase = "Artificial intelligence is metamorphosing the world!"
phrase2 = "GPT models use BPE tokenization to process unusual words like antidisestablishmentarianism."

# 1) tokens
tokens = tokenizer.tokenize(phrase)
print("Tokens:", tokens)

token_ids = tokenizer.encode(phrase)
print("Token IDs:", token_ids)

print("\nDétails par token:")
for tid in token_ids:
    txt = tokenizer.decode([tid])
    print(tid, repr(txt))
tokens2 = tokenizer.tokenize(phrase2)
print(tokens2)
long_word = " antidisestablishmentarianism" 
long_tokens = tokenizer.tokenize(long_word)

print("Sous-tokens (mot long):", long_tokens)
print("Nombre de sous-tokens:", len(long_tokens))