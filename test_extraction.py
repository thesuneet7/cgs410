#!/usr/bin/env python3
"""
Unit tests for the intervener extraction pipeline.
Run these BEFORE running on real data to verify correctness.

Tests cover:
  - CoNLL-U parsing
  - Intervener identification (the core formula)
  - Feature A: POS extraction
  - Feature B: Arity computation
  - Feature C: Subtree size computation
  - Feature D: Attachment type classification
  - Edge cases: root arcs, adjacent tokens, long sentences

Usage:
    python3 test_extraction.py
    python3 test_extraction.py -v   (verbose)
"""

import sys
import unittest

from conllu_parser import parse_conllu_string, Sentence
from step3_extract_interveners import extract_interveners_from_sentence


# -----------------------------------------------------------------------
# Sample CoNLL-U sentences for testing
# -----------------------------------------------------------------------

# Sentence: "John gave Mary a book"
# Tree:  gave(root) → John(nsubj), Mary(iobj), book(obj), .(punct)
#        book → a(det)
# Arc: gave(2) → John(1) → adjacent, no interveners
# Arc: gave(2) → Mary(3) → adjacent, no interveners
# Arc: gave(2) → book(5) → distance=3, interveners={3=Mary, 4=a}
# Arc: book(5) → a(4)    → adjacent, no interveners
SENTENCE_GAVE = """\
# sent_id = test-gave
# text = John gave Mary a book
1\tJohn\tjohn\tPROPN\tNNP\t_\t2\tnsubj\t_\t_
2\tgave\tgive\tVERB\tVBD\t_\t0\troot\t_\t_
3\tMary\tmary\tPROPN\tNNP\t_\t2\tiobj\t_\t_
4\ta\ta\tDET\tDT\t_\t5\tdet\t_\t_
5\tbook\tbook\tNOUN\tNN\t_\t2\tobj\t_\t_

"""

# Sentence: "The old man walked slowly"
# Tree:  walked(root) → man(nsubj), slowly(advmod)
#        man → The(det), old(amod)
# Arc: walked(4) → man(3) → adjacent, no interveners
# Arc: walked(4) → slowly(5) → adjacent, no interveners
# Arc: man(3) → The(1) → distance=2, intervener={2=old}
# Arc: man(3) → old(2) → adjacent
SENTENCE_WALKED = """\
# sent_id = test-walked
# text = The old man walked slowly
1\tThe\tthe\tDET\tDT\t_\t3\tdet\t_\t_
2\told\told\tADJ\tJJ\t_\t3\tamod\t_\t_
3\tman\tman\tNOUN\tNN\t_\t4\tnsubj\t_\t_
4\twalked\twalk\tVERB\tVBD\t_\t0\troot\t_\t_
5\tslowly\tslowly\tADV\tRB\t_\t4\tadvmod\t_\t_

"""

# Sentence with a complex intervener: "The book that I read was good"
# Arc: was(6) → book(2) → distance=4, interveners={3=that, 4=I, 5=read}
# Note: "read"(5) has children (nsubj=4=I), so arity=1 → is_head=1
# "that"(3) has no children in basic UD → arity=0
# "I"(4) has no children → arity=0
SENTENCE_RELATIVE = """\
# sent_id = test-relative
# text = The book that I read was good
1\tThe\tthe\tDET\tDT\t_\t2\tdet\t_\t_
2\tbook\tbook\tNOUN\tNN\t_\t6\tnsubj\t_\t_
3\tthat\tthat\tSCONJ\tWDT\t_\t5\tmark\t_\t_
4\tI\ti\tPRON\tPRP\t_\t5\tnsubj\t_\t_
5\tread\tread\tVERB\tVBD\t_\t2\tacl:relcl\t_\t_
6\twas\tbe\tAUX\tVBD\t_\t0\troot\t_\t_
7\tgood\tgood\tADJ\tJJ\t_\t6\txcomp\t_\t_

"""


# -----------------------------------------------------------------------
# Test cases
# -----------------------------------------------------------------------

class TestCoNLLUParser(unittest.TestCase):

    def test_basic_parse(self):
        sents = parse_conllu_string(SENTENCE_GAVE)
        self.assertEqual(len(sents), 1)
        s = sents[0]
        self.assertEqual(s.sent_id, "test-gave")
        self.assertEqual(len(s.tokens), 5)

    def test_token_fields(self):
        s = parse_conllu_string(SENTENCE_GAVE)[0]
        gave = s.tokens[1]  # 0-indexed list, idx=2
        self.assertEqual(gave.idx, 2)
        self.assertEqual(gave.form, "gave")
        self.assertEqual(gave.upos, "VERB")
        self.assertEqual(gave.head, 0)  # root
        self.assertEqual(gave.deprel, "root")

    def test_children_table(self):
        s = parse_conllu_string(SENTENCE_GAVE)[0]
        # gave(2) should have children: John(1), Mary(3), book(5)
        self.assertCountEqual(s.children[2], [1, 3, 5])
        # book(5) should have child: a(4)
        self.assertCountEqual(s.children[5], [4])
        # John(1) should have no children
        self.assertEqual(s.children.get(1, []), [])

    def test_arity(self):
        s = parse_conllu_string(SENTENCE_GAVE)[0]
        self.assertEqual(s.arity(2), 3)   # gave: John, Mary, book
        self.assertEqual(s.arity(5), 1)   # book: a
        self.assertEqual(s.arity(1), 0)   # John: leaf
        self.assertEqual(s.arity(4), 0)   # a: leaf

    def test_subtree_size(self):
        s = parse_conllu_string(SENTENCE_GAVE)[0]
        # gave dominates everything (root)
        self.assertEqual(s.subtree_size(2), 5)
        # book dominates itself + "a"
        self.assertEqual(s.subtree_size(5), 2)
        # John is a leaf
        self.assertEqual(s.subtree_size(1), 1)
        # a is a leaf
        self.assertEqual(s.subtree_size(4), 1)

    def test_attachment_type(self):
        s = parse_conllu_string(SENTENCE_GAVE)[0]
        # Arc: gave(2) → book(5), intervener at Mary(3)
        # Mary(3) has head = gave(2) = head_idx → "head"
        self.assertEqual(s.attachment_type(3, 2, 5), "head")
        # a(4) has head = book(5) = dep_idx → "dependent"
        self.assertEqual(s.attachment_type(4, 2, 5), "dependent")


class TestIntervenerIdentification(unittest.TestCase):
    """
    Test the core formula: interveners k where min(h,d) < k < max(h,d)
    """

    def test_no_interveners_adjacent(self):
        """Adjacent tokens have no interveners."""
        s = parse_conllu_string(SENTENCE_GAVE)[0]
        rows = extract_interveners_from_sentence(s, "English")
        # All rows should be for the arc gave(2)→book(5); no adjacent arc produces rows
        arc_distances = {r["arc_distance"] for r in rows}
        self.assertTrue(all(d > 1 for d in arc_distances))

    def test_correct_interveners_gave(self):
        """Arc gave(2)→book(5): interveners should be Mary(3) and a(4)."""
        s = parse_conllu_string(SENTENCE_GAVE)[0]
        rows = extract_interveners_from_sentence(s, "English")

        # Filter to the specific arc
        arc_rows = [r for r in rows if r["head_idx"] == 2 and r["dep_idx"] == 5]
        self.assertEqual(len(arc_rows), 2)

        intervener_forms = sorted(r["intervener_form"] for r in arc_rows)
        self.assertEqual(intervener_forms, ["Mary", "a"])

    def test_correct_interveners_walked(self):
        """Arc man(3)→The(1): intervener should be old(2)."""
        s = parse_conllu_string(SENTENCE_WALKED)[0]
        rows = extract_interveners_from_sentence(s, "English")

        arc_rows = [r for r in rows if r["head_idx"] == 3 and r["dep_idx"] == 1]
        self.assertEqual(len(arc_rows), 1)
        self.assertEqual(arc_rows[0]["intervener_form"], "old")

    def test_relative_clause_interveners(self):
        """Arc was(6)→book(2): interveners should be that(3), I(4), read(5)."""
        s = parse_conllu_string(SENTENCE_RELATIVE)[0]
        rows = extract_interveners_from_sentence(s, "English")

        arc_rows = [r for r in rows if r["head_idx"] == 6 and r["dep_idx"] == 2]
        self.assertEqual(len(arc_rows), 3)

        forms = sorted(r["intervener_form"] for r in arc_rows)
        self.assertEqual(forms, ["I", "read", "that"])


class TestFeatureA_POS(unittest.TestCase):

    def test_intervener_upos_recorded(self):
        """Feature A: POS of each intervener must be recorded correctly."""
        s = parse_conllu_string(SENTENCE_GAVE)[0]
        rows = extract_interveners_from_sentence(s, "English")

        # Arc gave(2)→book(5): Mary is PROPN, a is DET
        arc_rows = {r["intervener_form"]: r for r in rows
                    if r["head_idx"] == 2 and r["dep_idx"] == 5}

        self.assertEqual(arc_rows["Mary"]["intervener_upos"], "PROPN")
        self.assertEqual(arc_rows["a"]["intervener_upos"], "DET")

    def test_relative_clause_upos(self):
        """Various POS types in relative clause interveners."""
        s = parse_conllu_string(SENTENCE_RELATIVE)[0]
        rows = extract_interveners_from_sentence(s, "English")
        arc_rows = {r["intervener_form"]: r for r in rows
                    if r["head_idx"] == 6 and r["dep_idx"] == 2}

        self.assertEqual(arc_rows["that"]["intervener_upos"], "SCONJ")
        self.assertEqual(arc_rows["I"]["intervener_upos"], "PRON")
        self.assertEqual(arc_rows["read"]["intervener_upos"], "VERB")


class TestFeatureB_Arity(unittest.TestCase):

    def test_leaf_intervener_arity_zero(self):
        """Feature B: Leaf nodes have arity 0."""
        s = parse_conllu_string(SENTENCE_GAVE)[0]
        rows = extract_interveners_from_sentence(s, "English")

        # "a" (token 4) is a leaf — arity should be 0
        a_row = next(r for r in rows if r["intervener_form"] == "a")
        self.assertEqual(a_row["intervener_arity"], 0)
        self.assertEqual(a_row["is_head"], 0)

    def test_branching_intervener_arity(self):
        """Feature B: read(5) has two children: that(3)=mark + I(4)=nsubj → arity=2."""
        s = parse_conllu_string(SENTENCE_RELATIVE)[0]
        rows = extract_interveners_from_sentence(s, "English")

        arc_rows = {r["intervener_form"]: r for r in rows
                    if r["head_idx"] == 6 and r["dep_idx"] == 2}

        self.assertEqual(arc_rows["read"]["intervener_arity"], 2)  # that + I
        self.assertEqual(arc_rows["read"]["is_head"], 1)
        self.assertEqual(arc_rows["I"]["intervener_arity"], 0)
        self.assertEqual(arc_rows["I"]["is_head"], 0)


class TestFeatureC_SubtreeSize(unittest.TestCase):

    def test_leaf_subtree_size_one(self):
        """Feature C: Leaf nodes have subtree size 1."""
        s = parse_conllu_string(SENTENCE_GAVE)[0]
        rows = extract_interveners_from_sentence(s, "English")
        a_row = next(r for r in rows if r["intervener_form"] == "a")
        self.assertEqual(a_row["intervener_subtree_size"], 1)

    def test_complex_subtree(self):
        """Feature C: read(5) dominates itself + that(3) + I(4) = subtree size 3."""
        s = parse_conllu_string(SENTENCE_RELATIVE)[0]
        rows = extract_interveners_from_sentence(s, "English")
        arc_rows = {r["intervener_form"]: r for r in rows
                    if r["head_idx"] == 6 and r["dep_idx"] == 2}
        self.assertEqual(arc_rows["read"]["intervener_subtree_size"], 3)  # read+that+I
        self.assertEqual(arc_rows["I"]["intervener_subtree_size"], 1)


class TestFeatureD_Attachment(unittest.TestCase):

    def test_head_attached(self):
        """Feature D: Mary(3) depends on gave(2) = head → 'head' attachment."""
        s = parse_conllu_string(SENTENCE_GAVE)[0]
        rows = extract_interveners_from_sentence(s, "English")
        arc_rows = {r["intervener_form"]: r for r in rows
                    if r["head_idx"] == 2 and r["dep_idx"] == 5}
        self.assertEqual(arc_rows["Mary"]["attachment_type"], "head")

    def test_dep_attached(self):
        """Feature D: a(4) depends on book(5) = dep → 'dependent' attachment."""
        s = parse_conllu_string(SENTENCE_GAVE)[0]
        rows = extract_interveners_from_sentence(s, "English")
        arc_rows = {r["intervener_form"]: r for r in rows
                    if r["head_idx"] == 2 and r["dep_idx"] == 5}
        self.assertEqual(arc_rows["a"]["attachment_type"], "dependent")

    def test_external_attachment(self):
        """Feature D: read(5) depends on book(2) which is neither head(6) nor
        immediate dep — actually book IS dep here so read should be 'dependent'."""
        # Arc: was(6) → book(2). read(5) has head = book(2) = dep_idx → 'dependent'
        s = parse_conllu_string(SENTENCE_RELATIVE)[0]
        rows = extract_interveners_from_sentence(s, "English")
        arc_rows = {r["intervener_form"]: r for r in rows
                    if r["head_idx"] == 6 and r["dep_idx"] == 2}
        # read's head is book(2) which is dep_idx → dependent
        self.assertEqual(arc_rows["read"]["attachment_type"], "dependent")
        # that(3) has head read(5), which is neither 6 nor 2 → external
        self.assertEqual(arc_rows["that"]["attachment_type"], "external")


class TestMetadata(unittest.TestCase):

    def test_language_and_typology_recorded(self):
        """All metadata columns should be populated."""
        s = parse_conllu_string(SENTENCE_GAVE)[0]
        rows = extract_interveners_from_sentence(s, "English")
        for row in rows:
            self.assertEqual(row["language"], "English")
            self.assertEqual(row["word_order"], "SVO")
            self.assertIn(row["attachment_type"], ["head", "dependent", "external"])

    def test_arc_distance_correct(self):
        """arc_distance = |head_idx - dep_idx|"""
        s = parse_conllu_string(SENTENCE_GAVE)[0]
        rows = extract_interveners_from_sentence(s, "English")
        arc_rows = [r for r in rows if r["head_idx"] == 2 and r["dep_idx"] == 5]
        self.assertTrue(all(r["arc_distance"] == 3 for r in arc_rows))

    def test_num_interveners_correct(self):
        """num_interveners = count of tokens between h and d."""
        s = parse_conllu_string(SENTENCE_GAVE)[0]
        rows = extract_interveners_from_sentence(s, "English")
        arc_rows = [r for r in rows if r["head_idx"] == 2 and r["dep_idx"] == 5]
        self.assertTrue(all(r["num_interveners"] == 2 for r in arc_rows))


# -----------------------------------------------------------------------
# Run all tests
# -----------------------------------------------------------------------

if __name__ == "__main__":
    verbosity = 2 if "-v" in sys.argv else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(unittest.TestLoader().loadTestsFromModule(
        sys.modules[__name__]
    ))
    if result.wasSuccessful():
        print("\n[ALL TESTS PASSED] Extraction logic is correct.")
        print("Safe to run on real treebank data.")
    else:
        print(f"\n[FAILURES: {len(result.failures) + len(result.errors)}]")
        sys.exit(1)
