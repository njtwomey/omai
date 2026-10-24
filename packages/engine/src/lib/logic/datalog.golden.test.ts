import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { datalog, evaluateDatalog, leastModel, stableModels } from './index.ts'

describe('Datalog against clingo and a reference naive evaluator', () => {
  for (const c of loadFixture('logic/datalog').cases) {
    it(c.name, () => {
      const p = c.params
      const source = p['program'] as string
      if (c.fn === 'unstratifiable') {
        expect(() => datalog(source)).toThrow(/stratify/)
        expect(stableModels(source).models).toEqual(p['stableModels'])
        return
      }
      expect(leastModel(source)).toEqual(p['model'])
      for (const method of ['naive', 'semi-naive'] as const) {
        const r = evaluateDatalog(source, { method })
        const byStratum: number[][] = []
        for (const h of r.history) {
          while (byStratum.length <= h.stratum) byStratum.push([])
          byStratum[h.stratum]?.push(h.derived)
        }
        expect(byStratum).toEqual(p['rounds'])
      }
      // A stratified program has exactly one answer set: its perfect model.
      if (source.length < 400) expect(stableModels(source, { maxGuessAtoms: 20 }).models).toEqual([p['model']])
    })
  }
})

describe('answer sets against clingo', () => {
  for (const c of loadFixture('logic/answer-sets').cases) {
    it(c.name, () => {
      expect(stableModels(c.params['program'] as string).models).toEqual(c.params['models'])
    })
  }
})
