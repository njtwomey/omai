import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import { candidateElimination, findS, type Hypothesis, type LabelledExample, versionSpaceMembers } from './index.ts'

const show = (hs: readonly Hypothesis[]): string[][] => hs.map((h) => h.map((c) => c ?? '∅')).sort()
const sortKey = (a: string[], b: string[]): number => (JSON.stringify(a) < JSON.stringify(b) ? -1 : 1)

describe('version spaces against brute-force enumeration', () => {
  for (const c of loadFixture('logic/ilp/version-spaces').cases) {
    it(c.name, () => {
      const p = c.params
      const examples = p['examples'] as LabelledExample[]
      const domains = p['domains'] as string[][]
      const vs = candidateElimination(examples, domains)
      expect(show(vs.specific).sort(sortKey)).toEqual((p['specific'] as string[][]).sort(sortKey))
      expect(show(vs.general).sort(sortKey)).toEqual((p['general'] as string[][]).sort(sortKey))
      // The brute-force count includes the empty hypothesis when no positive has been seen.
      const members = versionSpaceMembers(vs).length + (vs.specific.some((h) => h.some((x) => x === null)) ? 1 : 0)
      expect(members).toBe(p['size'])
      expect(findS(examples)).toEqual(p['findS'])
    })
  }
})
