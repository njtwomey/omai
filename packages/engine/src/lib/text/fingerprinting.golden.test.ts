import { describe, expect, it } from 'vitest'
import { expectMatches, loadFixture } from '../../testing/golden.ts'
import { tensor } from '../tensor/index.ts'
import { lshCandidates, lshParameters, minhash, minhashJaccard, murmurhash3 } from './index.ts'

describe('text/fingerprinting against datasketch', () => {
  const cases = loadFixture('text/fingerprinting').cases
  const sigs = cases
    .filter((c) => c.fn === 'minhash')
    .map((c) => {
      const [items, a, b] = c.inputs as [string[], number[], number[]]
      return minhash(items, { a: Uint32Array.from(a), b: Uint32Array.from(b) }, (s) => murmurhash3(s, 0, true))
    })
  for (const c of cases) {
    it(c.name, () => {
      switch (c.fn) {
        case 'minhash':
          expect(Array.from(sigs[Number(c.name.split('/')[1])] ?? [])).toEqual(c.output)
          return
        case 'jaccard': {
          const est = sigs.slice(0, -1).map((s) => sigs.slice(0, -1).map((t) => minhashJaccard(s, t)))
          expectMatches(tensor(est), c.output, c.rtol, c.atol, c.name)
          return
        }
        case 'lshParameters': {
          const [threshold, perms] = c.inputs as [number, number]
          const p = lshParameters(threshold, perms)
          expect([p.bands, p.rows]).toEqual(c.output)
          return
        }
        default: {
          const pairs = lshCandidates(sigs, c.params['bands'] as number, c.params['rows'] as number)
          const neighbours = sigs.map((_, i) =>
            [...new Set([i, ...pairs.flatMap(([a, b]) => (a === i ? [b] : b === i ? [a] : []))])].sort((x, y) => x - y),
          )
          expect(neighbours).toEqual(c.output)
        }
      }
    })
  }
})
