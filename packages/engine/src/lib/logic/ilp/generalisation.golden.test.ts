import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import { parseClause, parseTerm, termToString } from '../index.ts'
import { isThetaEquivalent, lgg, rlgg } from './index.ts'

describe('generalisation against a reference port of anti-unification', () => {
  for (const c of loadFixture('logic/ilp/generalisation').cases) {
    it(c.name, () => {
      const p = c.params
      if (c.fn === 'lgg') {
        expect(termToString(lgg(parseTerm(p['a'] as string), parseTerm(p['b'] as string)))).toBe(p['lgg'])
        return
      }
      const r = rlgg(
        (p['examples'] as string[]).map((s) => parseTerm(s)),
        (p['background'] as string[]).map((s) => parseTerm(s)),
      )
      const expected = parseClause((p['clause'] as string).replace(/\bG(\d+)\b/g, 'X$1'))
      expect(isThetaEquivalent(r, expected)).toBe(true)
      expect(r.body.length).toBe(expected.body.length)
    })
  }
})
