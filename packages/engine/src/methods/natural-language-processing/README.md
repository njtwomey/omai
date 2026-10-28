# Natural language processing (part 7, application domains)

Text processed and modelled classically: text representation (tokenisation, stemming, vocabularies, bag of words, TF-IDF, hashing), language models (n-grams with smoothing, class-based), the NLP tasks (classification, named entities, parsing, similarity) and sequence labelling (hidden Markov models and conditional random fields for tagging), as sessions over `text`, `pgm`, `supervised-learning`'s models and `probabilistic-inference`'s HMM.

## Capabilities

- Tokenisation stepped: byte-pair encoding's merges one at a time on a corpus, the vocabulary growing; options: the corpus, the merge count.
- Vectorisation: the term-document matrix built, TF-IDF weights as a probe, a classifier on top with its most-weighted terms.
- n-gram language models: counts and smoothing, perplexity on held-out text, generation.
- Sequence labelling: an HMM tagger by Viterbi with the trellis as a probe, a CRF trained by gradient with its features' weights.
- Parsing: CKY's chart filling cell by cell on a small grammar.
- Topic models and embeddings link to `unsupervised-learning` and `representation-learning`.

## Examples to make

- BPE merges on a small corpus: the merge just made and the tokenisation of a sentence changing.
- The Viterbi trellis for a sentence, the best path emerging column by column.
- CKY filling its chart for a short sentence.
- A naive Bayes spam filter's most indicative words as training rows arrive.

## Rendering

- Tokens with tags (`tokens`), trellises and charts as heatmaps or boards, bars of term weights, trees for parses.
- Missing: a trellis panel with the chosen path; a parse-tree panel aligned with tokens.

## Shared

- `shared/`: the corpus slot (tiny corpora and grammars in `datasets`), the token probes.

## Problems

- `text/` moves here. CKY and CRF training are not in `lib` yet.
