import { describe, expect, it } from 'vitest'
import { expectMatches, loadFixture } from '../../../testing/golden.ts'
import { tensor } from '../../tensor/index.ts'
import * as P from '../index.ts'
import * as C from './index.ts'

const rowsOf = (p: Readonly<Record<string, unknown>>): P.learning.DataRow[] => p['data'] as P.learning.DataRow[]
/** Sort lists of names the way Python sorts lists (element by element, a prefix first). */
const byList = (a: readonly string[], b: readonly string[]): number => {
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const c = (a[i] as string) < (b[i] as string) ? -1 : (a[i] as string) > (b[i] as string) ? 1 : 0
    if (c !== 0) return c
  }
  return a.length - b.length
}
const sorted = (es: readonly (readonly string[])[]): string[][] => es.map((e) => [...e]).sort(byList)

describe('interventions against pgmpy', () => {
  for (const c of loadFixture('pgm/causal/interventions').cases) {
    it(c.name, () => {
      const bn = P.bayesNet(c.params['network'] as P.BayesNode[])
      const f = C.interventionalDistribution(
        bn,
        c.params['query'] as string[],
        c.params['do'] as Record<string, string>,
        c.params['evidence'] as Record<string, string>,
      )
      expectMatches(tensor(P.factorValues(f)), c.output, 1e-12, 1e-14, c.name)
    })
  }
})

describe('identification against pgmpy', () => {
  for (const c of loadFixture('pgm/causal/identification').cases) {
    it(c.name, () => {
      const g = P.dag(c.params['edges'] as [string, string][])
      const latent = c.params['latent'] as string[]
      const x = c.params['x'] as string
      const y = c.params['y'] as string
      expect(sorted(C.backdoorAdjustmentSets(g, x, y, { latent }))).toEqual(c.params['backdoor'])
      expect(sorted(C.frontdoorAdjustmentSets(g, x, y, { latent }))).toEqual(c.params['frontdoor'])
      const minimal = C.minimalAdjustmentSet(g, x, y, { latent })
      expect(minimal === null ? -1 : minimal.length).toBe(c.params['minimalSize'])
      if (minimal !== null) expect(C.isBackdoorAdjustmentSet(g, x, y, minimal) || minimal.length === 0).toBe(true)
    })
  }
})

describe('causal discovery against causal-learn', () => {
  for (const c of loadFixture('pgm/causal/discovery').cases) {
    it(c.name, () => {
      if (c.fn === 'ges') {
        const r = C.ges(rowsOf(c.params))
        expect(sorted(r.cpdag.directed)).toEqual(c.params['directed'])
        expect(sorted(r.cpdag.undirected)).toEqual(c.params['undirected'])
        return
      }
      expect(C.directLingam(rowsOf(c.params)).order).toEqual(c.params['order'])
    })
  }
})

describe('FCI against causal-learn', () => {
  for (const c of loadFixture('pgm/causal/fci').cases) {
    it(c.name, () => {
      const { pag } = C.fci(rowsOf(c.params), { significance: 0.05 })
      const normal = (e: { a: string; b: string; markA: string; markB: string }): string =>
        e.a < e.b ? `${e.a} ${e.markA}-${e.markB} ${e.b}` : `${e.b} ${e.markB}-${e.markA} ${e.a}`
      const theirs = (c.params['edges'] as { a: string; b: string; markA: string; markB: string }[]).map(normal).sort()
      expect(pag.edges.map(normal).sort()).toEqual(theirs)
    })
  }
})
