import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import {
  lemmatize,
  lexicon,
  MORPHOLOGICAL_SUBSTITUTIONS,
  morphy,
  type PorterMode,
  porterStem,
  porterStemTrace,
  type StopWordList,
  snowballStem,
  stopWords,
  type WordNetPos,
} from './index.ts'

describe('text/stem against nltk', () => {
  for (const c of loadFixture('text/stem').cases) {
    it(c.name, () => {
      const words = c.inputs[0] as string[]
      const mode = c.params['mode'] as PorterMode
      const got = words.map((w) => (c.fn === 'porterStem' ? porterStem(w, { mode }) : snowballStem(w)))
      const wrong = words.flatMap((w, i) =>
        got[i] === (c.output as string[])[i] ? [] : [[w, got[i], (c.output as string[])[i]]],
      )
      expect(wrong).toEqual([])
      if (c.fn === 'porterStem')
        for (const w of words) expect(porterStemTrace(w, { mode }).stem).toBe(porterStem(w, { mode }))
    })
  }
})

describe('text/lemmatise against nltk morphy', () => {
  const lex = lexicon(
    {
      n: ['dog', 'church', 'churche', 'bus', 'box', 'wolf', 'woman', 'pony', 'axis', 'axe', 'glass', 'fly', 'news'],
      v: ['run', 'be', 'go', 'fly', 'die', 'hope', 'hop', 'make', 'agree', 'agre', 'stop', 'use'],
      a: ['big', 'fast', 'late', 'good', 'nice'],
      r: ['fast', 'well'],
    },
    {
      n: { axes: ['axis', 'axe'], mice: ['mouse'] },
      v: { went: ['go'], was: ['be'], ran: ['run'], dying: ['die'] },
      a: { better: ['good'] },
      r: { better: ['well'] },
    },
  )
  for (const c of loadFixture('text/lemmatise').cases) {
    it(c.name, () => {
      if (c.fn === 'substitutions') {
        const ours = Object.fromEntries(
          Object.entries(MORPHOLOGICAL_SUBSTITUTIONS).map(([k, v]) => [k, v.map((r) => [...r])]),
        )
        expect(ours).toEqual(c.output)
        return
      }
      const [word, pos] = c.inputs as [string, WordNetPos]
      expect(c.fn === 'morphy' ? morphy(word, pos, lex) : lemmatize(word, { pos, lexicon: lex })).toEqual(c.output)
    })
  }
})

describe('text/stopwords against scikit-learn and gensim', () => {
  for (const c of loadFixture('text/stopwords').cases) {
    it(c.name, () => expect(stopWords(c.params['list'] as StopWordList)).toEqual(c.output))
  }
})
