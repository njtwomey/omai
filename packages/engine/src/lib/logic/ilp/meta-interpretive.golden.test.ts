import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import { clauseToString } from '../index.ts'
import { learnFromFailures, metagol } from './index.ts'

describe('meta-interpretive learning and learning from failures on Metagol and Popper tasks', () => {
  for (const c of loadFixture('logic/ilp/meta-interpretive').cases) {
    it(c.name, () => {
      const p = c.params
      const task = {
        background: p['background'] as string,
        predicates: p['predicates'] as string[],
        positive: p['positive'] as string[],
        negative: p['negative'] as string[],
        metarules: p['metarules'] as string[],
      }
      const expected = [...(p['program'] as string[])].sort()
      const m = metagol(task)
      expect(m.status).toBe('learned')
      expect(m.program.map((x) => clauseToString(x)).sort()).toEqual(expected)
      if (p['systems'] === 'both') {
        const head = (p['program'] as string[])[0]?.split('(')[0] ?? ''
        const recursive = (p['program'] as string[]).some((s) => s.split(':-')[1]?.includes(`${head}(`))
        const r = learnFromFailures({
          background: task.background,
          head: `${head}/2`,
          body: recursive ? [...task.predicates, `${head}/2`] : task.predicates,
          positive: task.positive,
          negative: task.negative,
        })
        expect(r.status).toBe('learned')
        expect(r.program.map((x) => clauseToString(x)).sort()).toEqual(expected)
      }
    })
  }
})
