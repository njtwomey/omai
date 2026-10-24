import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { type PunktParameters, sentenceSpans, splitSentences, trainPunkt } from './index.ts'
import { codePointSpans } from './tokenise.golden.test.ts'

describe('text/sentences against nltk Punkt', () => {
  const fixture = loadFixture('text/sentences')
  const train = fixture.cases.find((c) => c.fn === 'trainPunkt')
  const params = trainPunkt((train?.inputs[0] as string) ?? '')
  const rules: PunktParameters = {
    abbrevTypes: new Set(['mr', 'dr', 'jan', 'p.m', 'u.s']),
    collocations: new Set(),
    sentStarters: new Set(),
    orthoContext: new Map(),
  }
  for (const c of fixture.cases) {
    it(c.name, () => {
      if (c.fn === 'trainPunkt') {
        expect({
          abbrevTypes: [...params.abbrevTypes].sort(),
          collocations: [...params.collocations].sort(),
          sentStarters: [...params.sentStarters].sort(),
          orthoContext: [...params.orthoContext].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)),
        }).toEqual(c.output)
        return
      }
      const text = c.inputs[0] as string
      const which = c.params['params']
      const p = which === 'trained' ? params : which === 'rules' ? rules : undefined
      const spans = sentenceSpans(text, p, { realignBoundaries: c.params['realign'] !== false })
      expect(codePointSpans(text, spans)).toEqual(c.output)
      if (which === 'rules' && c.params['realign'] === undefined)
        expect(splitSentences(text, { abbreviations: ['Mr.', 'dr', 'jan', 'p.m', 'u.s'] })).toEqual(
          spans.map(([a, b]) => text.slice(a, b)),
        )
    })
  }
})
