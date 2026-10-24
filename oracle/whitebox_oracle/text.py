"""Helpers for the text case modules.

HF's BPE and WordPiece trainers number symbols carrying a continuing-subword prefix or an end-of-word suffix in
hash-map order, so when two pairs involving them tie for the most frequent, the merge HF makes depends on the hash
state of the process (and so on which case modules ran before). `tie_free_vocab_size` finds how far training can go
before the first tie whose resolution could depend on it, by replaying the frequency-greedy merges on the word
counts; fixtures stop there, which keeps them reproducible. (The replay only chooses the stopping point; the
reference results come from HF.)"""


def _symbols(word: str, prefix: str | None, suffix: str | None) -> list[str]:
    chars = list(word)
    out: list[str] = []
    for k, ch in enumerate(chars):
        s = ch
        if k > 0 and prefix is not None:
            s = prefix + s
        if k == len(chars) - 1 and suffix is not None:
            s = s + suffix
        out.append(s)
    return out


def tie_free_vocab_size(
    word_counts: dict[str, int], prefix: str | None, suffix: str | None, specials: int, limit: int
) -> int:
    """The vocabulary size (at most `limit`) reached when the next merge would be a tie not between plain characters."""
    words = [(_symbols(w, prefix, suffix), c) for w, c in word_counts.items()]
    vocab = {ch for w in word_counts for ch in w} | {s for syms, _ in words for s in syms}
    size = specials + len(vocab)
    while size < limit:
        pairs: dict[tuple[str, str], int] = {}
        for syms, c in words:
            for a, b in zip(syms, syms[1:], strict=False):
                pairs[(a, b)] = pairs.get((a, b), 0) + c
        if not pairs:
            return size
        best = max(pairs.values())
        tied = [p for p, c in pairs.items() if c == best]

        # Only ties between plain characters resolve the same way in every process (by code point).
        if len(tied) > 1 and not all(len(x) == 1 and len(y) == 1 for x, y in tied):
            return size
        a, b = min(tied)
        merged = a + (b[len(prefix) :] if prefix is not None and b.startswith(prefix) else b)
        if merged not in vocab:
            vocab.add(merged)
            size += 1
        new_words: list[tuple[list[str], int]] = []
        for syms, c in words:
            out: list[str] = []
            k = 0
            while k < len(syms):
                if k + 1 < len(syms) and syms[k] == a and syms[k + 1] == b:
                    out.append(merged)
                    k += 2
                else:
                    out.append(syms[k])
                    k += 1
            new_words.append((out, c))
        words = new_words
    return limit
