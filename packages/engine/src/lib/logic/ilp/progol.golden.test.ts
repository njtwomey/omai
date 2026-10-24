import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import { clauseToString, consult, parseTerm } from '../index.ts'
import { bottomClause, parseMode, progol } from './index.ts'

describe('inverse entailment on textbook tasks', () => {
  for (const c of loadFixture('logic/ilp/progol').cases) {
    it(c.name, () => {
      const p = c.params
      const modes = p['modes'] as string[]
      const [modeh, ...modeb] = modes
      const background = p['background'] as string
      const positive = p['positive'] as string[]
      const r = progol({ background, positive, negative: p['negative'] as string[], modeh: modeh ?? '', modeb })
      expect(r.theory.map((x) => clauseToString(x))).toEqual(p['theory'])
      const b = bottomClause(
        parseTerm(positive[0] ?? ''),
        consult(background),
        parseMode(modeh ?? ''),
        modeb.map(parseMode),
      )
      expect(clauseToString(b.clause)).toBe(p['bottom'])
    })
  }
})
