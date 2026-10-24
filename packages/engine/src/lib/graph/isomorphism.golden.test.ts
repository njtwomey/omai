import { describe, expect, it } from 'vitest'
import { decode, loadFixture } from '../../testing/golden.ts'
import { graphFromCase } from '../../testing/graphs.ts'
import type { Tensor } from '../tensor/index.ts'
import { isIsomorphic, isSubgraphIsomorphic, subgraphIsomorphisms, vf2, wlHash } from './index.ts'

describe('isomorphism against networkx', () => {
  for (const c of loadFixture('graph/isomorphism').cases) {
    it(c.name, () => {
      const p = c.params
      if (c.fn === 'wlHash' || c.fn === 'wlHashLabelled') {
        const graphs = (p['graphs'] as Record<string, unknown>[]).map((s) => graphFromCase(s))
        const labels = p['labels'] as string[][] | undefined
        const hashes = graphs.map((g, i) =>
          labels === undefined ? wlHash(g) : wlHash(g, { nodeLabels: labels[i] ?? [], iterations: 2 }),
        )
        const equal = hashes.map((h) => hashes.map((k) => h === k))
        expect(equal).toEqual(p['equal'])
        return
      }
      const host = graphFromCase(p)
      const pattern = graphFromCase(p, '2')
      if (c.fn === 'isomorphism') {
        expect(isIsomorphic(host, pattern)).toBe(p['isomorphic'])
        expect(Array.from(vf2(host, pattern)).length).toBe(p['count'])
        return
      }
      const [induced, mono] = Array.from((decode(c.output) as Tensor).data)
      const embeddings = subgraphIsomorphisms(host, pattern)
      expect(embeddings.length).toBe(induced)
      expect(new Set(embeddings.map((m) => m.join(','))).size).toBe(induced)
      expect(subgraphIsomorphisms(host, pattern, { induced: false }).length).toBe(mono)
      expect(isSubgraphIsomorphic(host, pattern)).toBe((induced ?? 0) > 0)
      expect(isSubgraphIsomorphic(host, pattern, { induced: false })).toBe((mono ?? 0) > 0)
    })
  }
})
