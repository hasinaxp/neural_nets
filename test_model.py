import torch

from tokenizer import Tokenizer
from simple_transformer import Transformer
from config import CONFIG

VOCAB_SIZE = CONFIG.get('vocab_size', 20000)
SEQ_LEN = 2024
EMBEDDING_DIM = CONFIG.get('embedding_dim', 512)
NUM_HEADS = 16
NUM_LAYERS = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOKENIZER_FILE = f"artifacts/tokenizer-{VOCAB_SIZE}.txt"
MODEL_FILE = "artifacts/pretrain_model.pt"

DEFAULT_MAX_COUNT = 200

print(f"Using device: {DEVICE}")

print("Loading tokenizer...")
tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
tokenizer.load(TOKENIZER_FILE)
actual_vocab_size = max(tokenizer.vocab) + 1 if tokenizer.vocab else tokenizer.vocab_size

print("Loading model...")
model = Transformer(
    vocab_size=actual_vocab_size,
    n_layer=NUM_LAYERS,
    n_head=NUM_HEADS,
    n_dim=EMBEDDING_DIM,
    n_seq=SEQ_LEN,
).to(DEVICE)

state_dict = torch.load(MODEL_FILE, map_location=DEVICE)
if "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]
model.load_state_dict(state_dict)
model.eval()

print(f"Model loaded from {MODEL_FILE} ({model.get_param_count():,} params)")


def generate_text(prompt, max_count=DEFAULT_MAX_COUNT, **generate_kwargs):
    bos = tokenizer.special_tokens["<|BOS|>"]
    prompt_tokens = tokenizer.encode(prompt) if prompt else []
    input_tokens = [bos] + prompt_tokens
    input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        generated = model.generate(input_tensor, max_count=max_count, **generate_kwargs)

    generated_tokens = generated[0].cpu().tolist()
    return tokenizer.decode(generated_tokens)


HELP_TEXT = """
Commands:
  /help            show this message
  /maxlen N        set max generated token count (current: {max_count})
  /exit, /quit     leave the REPL
Anything else is treated as a prompt and sent to the model.
"""


def repl():
    max_count = DEFAULT_MAX_COUNT
    print(HELP_TEXT.format(max_count=max_count))

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in ("/exit", "/quit"):
            break
        if line == "/help":
            print(HELP_TEXT.format(max_count=max_count))
            continue
        if line.startswith("/maxlen"):
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                max_count = int(parts[1])
                print(f"max_count set to {max_count}")
            else:
                print("usage: /maxlen N")
            continue

        try:
            output = generate_text(line, max_count=max_count)
        except Exception as e:
            print(f"generation failed: {e}")
            continue

        print(output)


if __name__ == "__main__":
    repl()