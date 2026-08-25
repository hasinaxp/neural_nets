import os
from collections import Counter
from tqdm import tqdm


class Tokenizer:
    """
    - Ascii-only
    - Sentence-level sub-word BPE tokenizer: merges are learned, and applied at
      encode time, independently per sentence — a merge never bridges across a
      sentence-terminal ('.', '!', '?') boundary.
    - Special tokens: <|BOS|>, <|EOS|>, <|PAD|>, <|UNK|>, <|USER|>, <|ASSISTANT|>
    """
    DEFAULT_VOCAB = {' ': 0, '\n': 1, '\t': 2}

    # Separator used in the on-disk vocab format. Doesn't need escaping itself as
    # long as escaped tokens never happen to contain this exact literal substring,
    # which is astronomically unlikely for ASCII/BPE-merged tokens.
    _SEP = "|_@:/_|"

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = self.DEFAULT_VOCAB.copy()
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merges = []          # list of (id1, id2, new_id)
        self.current_vocab_size = len(self.vocab)

    # ------------------------------------------------------------------
    # ASCII filtering
    # ------------------------------------------------------------------
    @staticmethod
    def _ascii_only(text: str) -> str:
        """Drop any non-ASCII character entirely (not mapped to <|UNK|>, just removed)."""
        return ''.join(c for c in text if ord(c) < 128)

    # ------------------------------------------------------------------
    # Sentence segmentation
    # ------------------------------------------------------------------
    def _split_segments(self, text: str) -> list[str]:
        """
        Split text into segments right after each sentence-terminal character
        ('.', '!', '?'), WITHOUT reformatting whitespace. Concatenating the
        returned segments reproduces `text` exactly — this is what lets
        `encode`/`decode` stay lossless while still respecting sentence
        boundaries the same way `train_from_file` does.
        """
        segments = []
        start = 0
        for i, ch in enumerate(text):
            if ch in ('.', '!', '?'):
                segments.append(text[start:i + 1])
                start = i + 1
        if start < len(text):
            segments.append(text[start:])
        return segments

    def _apply_merges(self, token_ids: list[int]) -> list[int]:
        """Apply learned merges (in order) to a single token-id sequence."""
        for id1, id2, new_id in self.merges:
            new_tokens = []
            i = 0
            while i < len(token_ids):
                if i + 1 < len(token_ids) and token_ids[i] == id1 and token_ids[i + 1] == id2:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(token_ids[i])
                    i += 1
            token_ids = new_tokens
        return token_ids

    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs using learned BPE merges. Text is first split
        into sentence-level segments (matching how merges were trained), and
        merges are applied independently within each segment so no merge can
        bridge across a sentence boundary.
        """
        unk_id = self.vocab.get('<|UNK|>')
        if unk_id is None:
            raise ValueError("<|UNK|> token not found in vocabulary")

        text = self._ascii_only(text)

        all_tokens = []
        for segment in self._split_segments(text):
            seg_tokens = [self.vocab.get(char, unk_id) for char in segment]
            all_tokens.extend(self._apply_merges(seg_tokens))
        return all_tokens

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to text, using <|UNK|> for unknown IDs."""
        unk_str = '<|UNK|>'
        return ''.join(self.id_to_token.get(t, unk_str) for t in tokens)

    def train_from_file(self, file_path: str):
        """Train BPE from an ASCII text file using sentence-level in-place merges."""
        # Read file and keep only ASCII
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        text = self._ascii_only(text)

        # Build base vocabulary: unique characters + special tokens
        self.vocab = self.DEFAULT_VOCAB.copy()
        for char in set(text):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        special_tokens = [
            '<|BOS|>', '<|EOS|>', '<|PAD|>',
            '<|UNK|>', '<|USER|>', '<|ASSISTANT|>'
        ]
        for token in special_tokens:
            self.vocab[token] = len(self.vocab)

        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merges = []

        # Tokenize each sentence into a list of character token IDs
        sentences = self.get_sentences(text)
        sentence_tokens = []
        for s in sentences:
            tokens = [self.vocab[char] for char in s]   # all chars are in vocab
            if tokens:
                sentence_tokens.append(tokens)

        # BPE training loop
        initial_vocab_size = len(self.vocab)
        pbar = tqdm(total=self.vocab_size - initial_vocab_size,
                    desc="BPE training", unit="merge")

        while len(self.vocab) < self.vocab_size:
            # Count adjacent pairs across all sentences (never across sentences)
            pair_counts = Counter()
            for seq in sentence_tokens:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += 1

            if not pair_counts:
                break

            best_pair_ids, freq = pair_counts.most_common(1)[0]
            if freq < 2:   # no pair appears often enough to merge
                break

            id1, id2 = best_pair_ids
            t1 = self.id_to_token[id1]
            t2 = self.id_to_token[id2]
            merged_token = t1 + t2
            new_id = len(self.vocab)

            # Update vocabulary and merge history (as integer triple)
            self.vocab[merged_token] = new_id
            self.id_to_token[new_id] = merged_token
            self.merges.append((id1, id2, new_id))

            # Replace the pair in every sentence (in-place)
            for idx, seq in enumerate(sentence_tokens):
                new_seq = []
                i = 0
                while i < len(seq):
                    if i + 1 < len(seq) and seq[i] == id1 and seq[i + 1] == id2:
                        new_seq.append(new_id)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                sentence_tokens[idx] = new_seq

            pbar.update(1)

        pbar.close()
        self.current_vocab_size = len(self.vocab)

    # ------------------------------------------------------------------
    # Serialization. Tokens are escaped so that '\\', '|', '@', '\n', and '\r'
    # can never corrupt the line-based file format on save/load.
    # ------------------------------------------------------------------
    @staticmethod
    def _escape_token(token: str) -> str:
        # Order matters: escape backslashes first so the backslashes introduced
        # for \n/\r below aren't themselves re-escaped.
        token = token.replace("\\", "\\\\")
        token = token.replace("|", "||").replace("@", "@@")
        token = token.replace("\n", "\\n").replace("\r", "\\r")
        return token

    @staticmethod
    def _unescape_token(token: str) -> str:
        # Reverse order of _escape_token.
        token = token.replace("\\n", "\n").replace("\\r", "\r")
        token = token.replace("||", "|").replace("@@", "@")
        token = token.replace("\\\\", "\\")
        return token

    def save(self, file_path: str):
        """Save vocabulary and merge rules to a file (integer triples for merges)."""
        with open(file_path, "w", encoding="utf-8") as f:
            for token, idx in self.vocab.items():
                escaped_token = self._escape_token(token)
                f.write(f"{escaped_token}{self._SEP}{idx}\n")
            f.write("MERGES\n")
            for id1, id2, new_id in self.merges:
                f.write(f"{id1} {id2} {new_id}\n")

    def load(self, file_path: str):
        """Load vocabulary and merge rules from a saved file."""
        self.vocab = self.DEFAULT_VOCAB.copy()
        self.merges = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Read vocab until "MERGES" marker
        merge_marker = "MERGES\n"
        idx = 0
        while idx < len(lines) and lines[idx] != merge_marker:
            line = lines[idx].rstrip("\n")
            if not line:
                idx += 1
                continue
            parts = line.split(self._SEP)
            if len(parts) != 2:
                idx += 1
                continue
            token, idx_str = parts
            token = self._unescape_token(token)
            try:
                self.vocab[token] = int(idx_str)
            except ValueError:
                pass
            idx += 1

        if idx < len(lines) and lines[idx] == merge_marker:
            idx += 1

        # Read merges as integer triples
        while idx < len(lines):
            line = lines[idx].strip()
            if not line:
                idx += 1
                continue
            parts = line.split()
            if len(parts) != 3:
                idx += 1
                continue
            try:
                id1, id2, new_id = int(parts[0]), int(parts[1]), int(parts[2])
                self.merges.append((id1, id2, new_id))
            except ValueError:
                pass
            idx += 1

        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.current_vocab_size = len(self.vocab)

    def get_sentences(self, text: str) -> list[str]:
        """Split ASCII text into sentences (training-corpus cleanup: collapses
        hard-wrapped lines within a paragraph into single lines). Used only for
        training — `encode` uses the lossless `_split_segments` instead so that
        encode/decode round-trips exactly."""
        text = text.strip()
        text = self._ascii_only(text)
        blocks = text.split('\n\n')
        for i in range(len(blocks)):
            lines = [line.strip() for line in blocks[i].splitlines() if line.rstrip()]
            blocks[i] = ' '.join(lines)
        text = '\n\n'.join(blocks)
        sentences = []
        current = []
        for char in text:
            current.append(char)
            if char in ('.', '!', '?'):
                sentences.append(''.join(current).strip())
                current = []
        if current:
            sentences.append(''.join(current).strip())
        return sentences


# test
if __name__ == "__main__":
    tokenizer = Tokenizer(vocab_size=1000)        # small value for quick testing
    tokenizer.train_from_file("datasets/sample.txt")      # ensure sample.txt exists

    text = """
Away, you fool! it more becomes a man
Than gilt his trophy: the breasts of Hecuba,
When she did suckle Hector, look'd not lovelier
Than Hector's forehead when it spit forth blood
At Grecian sword, contemning. Tell Valeria,
We are fit to bid her welcome.
    """
    tokens = tokenizer.encode(text)
    print(f"Encoded: {tokens}, length: {len(tokens)}, char-token compression: {len(text)/len(tokens):.2f}")
    decoded_text = tokenizer.decode(tokens)
    print(f"Decoded: {decoded_text}")