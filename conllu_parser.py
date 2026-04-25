"""
conllu_parser.py — Pure-Python CoNLL-U parser (no external dependencies)
Project: Empirical Distribution of Intervener Complexity

CoNLL-U format (10 tab-separated fields per token):
  1  ID      - Word index (integer). Multi-word: 1-2. Empty node: 1.1
  2  FORM    - Word form / punctuation symbol
  3  LEMMA   - Lemma or stem
  4  UPOS    - Universal POS tag
  5  XPOS    - Language-specific POS tag
  6  FEATS   - Morphological features (key=value pairs, | separated)
  7  HEAD    - Head of current word (integer, 0 = root)
  8  DEPREL  - Universal dependency relation
  9  DEPS    - Enhanced dependency graph
  10 MISC    - Any other annotation

Reference: https://universaldependencies.org/format.html

This parser:
  - Skips multi-word tokens (IDs like "1-2")
  - Skips empty nodes (IDs like "1.1")
  - Handles the full UD v2.x format
  - Returns sentences as lists of Token namedtuples for easy access
"""

from typing import Iterator, List, Optional
from dataclasses import dataclass, field


# -----------------------------------------------------------------------
# Token data structure
# -----------------------------------------------------------------------
@dataclass
class Token:
    """Represents one token in a CoNLL-U sentence."""
    idx: int            # 1-based linear position in sentence (our own counter)
    id: str             # Original CoNLL-U ID field (could be "1", "1-2", "1.1")
    form: str           # Surface word form
    lemma: str          # Lemma
    upos: str           # Universal POS tag  ← the main field we use
    xpos: str           # Language-specific POS
    feats: str          # Morphological features
    head: int           # Head index (0 = root). -1 if not a regular token
    deprel: str         # Dependency relation label
    deps: str           # Enhanced dependencies
    misc: str           # Misc field


@dataclass
class Sentence:
    """Represents one parsed sentence."""
    sent_id: str               # From # sent_id = ... comment
    text: str                  # From # text = ... comment
    tokens: List[Token]        # Only regular tokens (no MWT or empty nodes)
    comments: List[str]        # All raw comment lines

    # Precomputed structures for fast lookup
    # children[i] = list of token indices (1-based) whose head is i
    children: dict = field(default_factory=dict)

    def __post_init__(self):
        """Build the children lookup table after init."""
        self.children = {tok.idx: [] for tok in self.tokens}
        self.children[0] = []  # root has no parent but may have children index 0
        for tok in self.tokens:
            if tok.head >= 0:
                if tok.head not in self.children:
                    self.children[tok.head] = []
                self.children[tok.head].append(tok.idx)

    def get_token(self, idx: int) -> Optional[Token]:
        """Get token by 1-based position index. Returns None if not found."""
        if 1 <= idx <= len(self.tokens):
            return self.tokens[idx - 1]
        return None

    def subtree_size(self, idx: int) -> int:
        """
        Compute the subtree size rooted at token `idx`.
        Subtree size = number of tokens dominated by this node (including itself).
        Uses iterative BFS to avoid recursion limit issues on deep trees.

        This is Feature C from our paper: phrase length of the intervening material.
        (Yadav et al. 2022 use syntactic heads; we additionally compute full subtree size)
        """
        visited = set()
        queue = [idx]
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            for child in self.children.get(current, []):
                queue.append(child)
        return len(visited)

    def arity(self, idx: int) -> int:
        """
        Compute the arity of token `idx`.
        Arity = number of DIRECT syntactic children in the UD tree.

        This is Feature B from our paper.
        A leaf node has arity 0. A verb with subject + object has arity 2.
        """
        return len(self.children.get(idx, []))

    def attachment_type(self, intervener_idx: int, head_idx: int, dep_idx: int) -> str:
        """
        Determine whether an intervening token is structurally attached to:
          - 'head'      : its own head in the tree is the arc head (h)
          - 'dependent' : its own head in the tree is the arc dependent (d)
          - 'external'  : its own head is neither h nor d

        This is Feature D — the attachment analysis from our proposal.
        Implements the classification described in Yadav et al. (2022)
        and extended in our research proposal.
        """
        tok = self.get_token(intervener_idx)
        if tok is None:
            return "external"
        if tok.head == head_idx:
            return "head"
        elif tok.head == dep_idx:
            return "dependent"
        else:
            return "external"


# -----------------------------------------------------------------------
# Parser
# -----------------------------------------------------------------------
def parse_conllu_file(filepath: str) -> Iterator[Sentence]:
    """
    Generator that yields one Sentence object per sentence in a .conllu file.

    Usage:
        for sentence in parse_conllu_file("en_ewt-ud-train.conllu"):
            for token in sentence.tokens:
                print(token.upos, token.head)

    Notes:
        - Multi-word tokens (e.g. ID "1-2") are skipped — they are surface
          forms only; the syntactic structure uses individual tokens.
        - Empty nodes (e.g. ID "1.1") are skipped — they are for enhanced
          dependencies only and not part of basic UD trees.
        - Tokens with HEAD = 0 are roots. We store head=0 for these.
        - Underscore ("_") values in UPOS or HEAD are treated as missing.
    """
    comments = []
    token_buffer = []
    sent_id = ""
    text = ""

    # We track a sequential 1-based index for regular tokens only
    # (separate from the CoNLL-U ID because MWT tokens shift the mapping)
    token_seq = 0

    def flush_sentence():
        """Convert buffer to Sentence and reset."""
        nonlocal token_seq, sent_id, text
        if not token_buffer:
            return None
        sent = Sentence(
            sent_id=sent_id,
            text=text,
            tokens=list(token_buffer),
            comments=list(comments),
        )
        token_seq = 0
        sent_id = ""
        text = ""
        comments.clear()
        token_buffer.clear()
        return sent

    with open(filepath, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")

            # --- Blank line = sentence boundary ---
            if line == "":
                sent = flush_sentence()
                if sent is not None:
                    yield sent
                continue

            # --- Comment lines ---
            if line.startswith("#"):
                comments.append(line)
                if line.startswith("# sent_id"):
                    sent_id = line.split("=", 1)[-1].strip()
                elif line.startswith("# text"):
                    text = line.split("=", 1)[-1].strip()
                continue

            # --- Token line ---
            fields = line.split("\t")
            if len(fields) != 10:
                # Malformed line — skip silently
                continue

            raw_id = fields[0]

            # Skip multi-word tokens (e.g. "1-2") and empty nodes (e.g. "1.1")
            if "-" in raw_id or "." in raw_id:
                continue

            # Parse HEAD — underscore means missing (treat as -1)
            try:
                head_val = int(fields[6])
            except ValueError:
                head_val = -1  # missing or "_"

            # Parse UPOS — underscore means unknown
            upos = fields[3] if fields[3] != "_" else "X"

            token_seq += 1
            tok = Token(
                idx=token_seq,
                id=raw_id,
                form=fields[1],
                lemma=fields[2],
                upos=upos,
                xpos=fields[4],
                feats=fields[5],
                head=head_val,
                deprel=fields[7],
                deps=fields[8],
                misc=fields[9],
            )
            token_buffer.append(tok)

    # Flush last sentence if file doesn't end with blank line
    sent = flush_sentence()
    if sent is not None:
        yield sent


def parse_conllu_string(text: str) -> List[Sentence]:
    """
    Parse CoNLL-U formatted text from a string (useful for testing).
    Returns a list of Sentence objects.
    """
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".conllu",
                                     encoding="utf-8", delete=False) as f:
        f.write(text)
        tmppath = f.name
    try:
        return list(parse_conllu_file(tmppath))
    finally:
        os.unlink(tmppath)


# -----------------------------------------------------------------------
# Quick self-test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    SAMPLE = """# sent_id = test-1
# text = John ate a mango
1\tJohn\tjohn\tPROPN\tNNP\t_\t2\tnsubj\t_\t_
2\tate\teat\tVERB\tVBD\t_\t0\troot\t_\t_
3\ta\ta\tDET\tDT\t_\t4\tdet\t_\t_
4\tmango\tmango\tNOUN\tNN\t_\t2\tobj\t_\t_
5\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_

"""
    sents = parse_conllu_string(SAMPLE)
    s = sents[0]
    print(f"Sentence: {s.text}")
    print(f"Tokens:   {[t.form for t in s.tokens]}")
    print(f"Children of token 2 (ate): {s.children[2]}")
    print(f"Arity of token 2 (ate): {s.arity(2)}")
    print(f"Subtree size of token 4 (mango): {s.subtree_size(4)}")
    print(f"Subtree size of token 2 (ate): {s.subtree_size(2)}")
    print("Parser self-test PASSED")
