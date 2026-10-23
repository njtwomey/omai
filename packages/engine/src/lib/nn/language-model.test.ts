import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import {
  bigramLanguageModel,
  characterTokenizer,
  languageModelFromModule,
  nextTokenLogits,
  perplexity,
  sequenceLogProb,
} from './language-model.ts'
import { Embedding } from './module/layers.ts'

describe('language models', () => {
  it('a character tokenizer round-trips text and places special tokens after the alphabet', () => {
    const tok = characterTokenizer('xyz', { bos: true, eos: true, pad: true })
    expect(tok.vocabSize).toBe(6)
    expect([tok.bosId, tok.eosId, tok.padId]).toEqual([3, 4, 5])
    expect(tok.decode([3, ...tok.encode('zyx'), 4, 5])).toBe('zyx')
    expect(() => tok.encode('w')).toThrow()
    expect(() => characterTokenizer('aa')).toThrow()
  })

  it('a bigram model scores each position by the current token, also through its cache', () => {
    const table = [
      [0, 1, 2],
      [3, 4, 5],
      [6, 7, 8],
    ]
    const lm = bigramLanguageModel(table)
    expect(toFlat(lm.logits([[2, 0]]))).toEqual([6, 7, 8, 0, 1, 2])
    expect(toFlat(lm.logits(tensor([[1]], { dtype: 'int32' })))).toEqual([3, 4, 5])
    const cache = lm.initCache?.(1) ?? null
    expect(toFlat(lm.step?.([[1]], cache).logits ?? tensor([]))).toEqual([3, 4, 5])
    expect(nextTokenLogits(lm, [0, 2])).toEqual([6, 7, 8])
    expect(() => lm.logits([[0], [1, 2]])).toThrow()
    expect(() => bigramLanguageModel([[0, 1]])).toThrow()
    expect(() => nextTokenLogits(lm, [])).toThrow()
  })

  it('log-probabilities and perplexity', () => {
    const lm = bigramLanguageModel([
      [0, Math.log(3)],
      [0, 0],
    ])
    // p(1 | 0) = 3/4, p(0 | 1) = 1/2.
    expect(sequenceLogProb(lm, [0, 1, 0])).toBeCloseTo(Math.log(0.75) + Math.log(0.5), 12)
    expect(sequenceLogProb(lm, [0])).toBe(0)
    expect(perplexity(lm, [0, 1, 0])).toBeCloseTo(Math.exp(-(Math.log(0.75) + Math.log(0.5)) / 2), 12)
    expect(() => perplexity(lm, [0])).toThrow()
  })

  it('wraps a module mapping ids to logits', () => {
    const tok = characterTokenizer('abcd')
    const lm = languageModelFromModule(new Embedding(4, 4, { key: key(0) }), tok, { maxLength: 3 })
    expect(lm.logits([[0, 1, 2]]).shape).toEqual([1, 3, 4])
    expect(() => lm.logits([[0, 1, 2, 3]])).toThrow()
    const bad = languageModelFromModule(new Embedding(4, 2, { key: key(0) }), tok)
    expect(() => bad.logits([[0]])).toThrow()
    const withEmbed = languageModelFromModule(new Embedding(4, 4, { key: key(0) }), tok, { embed: (t) => t })
    expect(withEmbed.embed?.([[1, 2]]).shape).toEqual([1, 2])
  })
})
