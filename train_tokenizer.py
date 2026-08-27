import os
from pretrain_dataset import load_prcessed_wikipedia_chunks
from tokenizer import Tokenizer
from config import CONFIG


def train_tokenizer(
    vocab_size = CONFIG.get('vocab_size', 32_768) 
):
    wiki_texts = '\n'.join(load_prcessed_wikipedia_chunks(1024 * 1024))
    python_code = ''
    python_code += open('simple_transformer.py').read()
    python_code += open('tokenizer.py').read()
    python_code += open('download_fineweb.py').read()
    print("corpus loaded")
    tokenizer = Tokenizer(vocab_size)
    tokenizer.train_from_text(wiki_texts + python_code)
    tokenizer.save(f'artifacts/tokenizer-{vocab_size}.txt')

if __name__ == '__main__':
    train_tokenizer()