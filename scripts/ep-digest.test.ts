import { describe, expect, it } from 'vitest'
import {
  compact,
  compare,
  computeDigest,
  type Digest,
  diffPaths,
  digestText,
  type Item,
  type J,
} from './ep-digest-core.ts'

// The digest machinery on three atoms (the full run is `make ep-digest`): two runs agree to the bit, and a perturbed digest is
// caught at the right level. No coverage rows (they need every atom), so the items are the atoms' and their variational runs'.
const only = ['atom/gaussian', 'atom/beta-bernoulli', 'atom/gamma-poisson']

/** A copy of a digest with one item replaced by `f` of it. */
function altered(d: Digest, key: string, f: (it: Item) => Item): Digest {
  return { ...d, items: { ...d.items, [key]: f(d.items[key] as Item) } }
}

/** The fields of an atom's item that the tests move. */
interface Parts {
  exact: { graphs: { ep: { hash: string } }; sweeps: number }
  num: { evidence: number }
}

describe('ep-digest', () => {
  it('is the same on two runs, byte for byte, and covers the atoms, their variational runs and the registry', async () => {
    const a = await computeDigest({ only, coverage: false })
    const b = await computeDigest({ only, coverage: false })
    expect(digestText(compact(a.digest))).toBe(digestText(compact(b.digest)))
    expect([...a.graphs.entries()]).toEqual([...b.graphs.entries()])
    // a prefix chooses the atom and its learned twin; each has its variational run
    const keys = Object.keys(a.digest.items)
    expect(keys).toHaveLength(12)
    for (const k of ['atom/gaussian', 'atom/gaussian-learned', 'vmp/beta-bernoulli', 'vmp/gamma-poisson-learned'])
      expect(keys).toContain(k)
    const registry = a.digest.registry as { operatorNames: string[]; operators: { name: string; hooks: string[] }[] }
    expect(registry.operatorNames.length).toBeGreaterThan(60)
    const probit = registry.operators.find((o) => o.name === 'probit')
    expect(probit?.hooks).toContain('ep')
    // each atom has its graphs in three compilations, hashed, and the full JSON in the store
    const gaussian = a.digest.items['atom/gaussian'] as Item
    const graphs = (gaussian.exact as unknown as { graphs: Record<string, { hash: string }> }).graphs
    expect(Object.keys(graphs)).toEqual(['ep', 'rows', 'vmp'])
    expect(graphs['ep']?.hash).toMatch(/^[0-9a-f]{64}$/)
    expect(a.graphs.has('atom/gaussian.ep')).toBe(true)
    const base = compact(a.digest)
    const same = compare(base, compact(b.digest), undefined, undefined, false)
    expect(same.level).toBe('bits')
    expect(same.moves).toEqual([])
  }, 60_000)

  it('sees a value that moved, at the level of the move', async () => {
    const { digest } = await computeDigest({ only, coverage: false })
    const base = compact(digest)
    const key = 'atom/gaussian'
    const bump = (rel: number) =>
      compare(
        base,
        altered(base, key, (it) => {
          const num = JSON.parse(JSON.stringify(it.num)) as { beliefs: { mu: { mean: number } } }
          num.beliefs.mu.mean *= 1 + rel
          return { ...it, num: num as unknown as J }
        }),
        undefined,
        undefined,
        false,
      )
    // inside the tolerance (1e-9 relative): the tolerant level, with the move listed
    const small = bump(1e-11)
    expect(small.level).toBe('tolerant')
    expect(small.moves.map((m) => `${m.item} ${m.path}`)).toEqual([`${key} num.beliefs.mu.mean`])
    // outside it: a failure that names the item, the field, the old and the new value
    const large = bump(1e-6)
    expect(large.level).toBe('none')
    expect(large.moves[0]).toMatchObject({ item: key, path: 'num.beliefs.mu.mean', level: 'number' })
    expect(large.moves[0]?.was).not.toBe(large.moves[0]?.now)
  }, 60_000)

  it('fails on a graph hash, a sweep count and an evidence that moved, and on a missing item', async () => {
    const { digest } = await computeDigest({ only, coverage: false })
    const base = compact(digest)
    const key = 'atom/beta-bernoulli'
    const edit = (f: (p: Parts) => void): Digest =>
      altered(base, key, (it) => {
        const parts = JSON.parse(JSON.stringify({ exact: it.exact, num: it.num })) as Parts
        f(parts)
        return { ...it, exact: parts.exact as unknown as J, num: parts.num as unknown as J }
      })
    const hash = compare(
      base,
      edit((p) => {
        p.exact.graphs.ep.hash = '0'.repeat(64)
      }),
      undefined,
      undefined,
      false,
    )
    expect(hash.level).toBe('none')
    expect(hash.moves).toMatchObject([{ item: key, path: 'exact.graphs.ep.hash', level: 'exact' }])
    const sweeps = compare(
      base,
      edit((p) => {
        p.exact.sweeps += 1
      }),
      undefined,
      undefined,
      false,
    )
    expect(sweeps.level).toBe('none')
    const evidence = compare(
      base,
      edit((p) => {
        p.num.evidence += 1e-10
      }),
      undefined,
      undefined,
      false,
    )
    expect(evidence.level).toBe('none')
    const tinyEvidence = compare(
      base,
      edit((p) => {
        p.num.evidence += 1e-13
      }),
      undefined,
      undefined,
      false,
    )
    expect(tinyEvidence.level).toBe('tolerant')
    const gone: Digest = { ...base, items: Object.fromEntries(Object.entries(base.items).filter(([k]) => k !== key)) }
    expect(compare(base, gone, undefined, undefined, false).moves[0]).toMatchObject({ item: key, path: '(item)' })
  }, 60_000)

  it('shows a changed graph as what moved in it: factors and variables matched by id and name', () => {
    const a: J = {
      variables: [
        { name: 'x', family: 'gaussian' },
        { name: 'y', family: 'gaussian' },
      ],
      factors: [
        { id: 'f0', operator: 'prior', params: { eta: [0, 1] } },
        { id: 'f1', operator: 'gaussian' },
      ],
    }
    const b: J = {
      variables: [
        { name: 'y', family: 'gaussian' },
        { name: 'x', family: 'gamma' },
      ],
      factors: [{ id: 'f0', operator: 'prior', params: { eta: [0, 2] } }],
    }
    const lines = diffPaths(a, b)
    expect(lines).toContain('variables[x].family: gaussian -> gamma')
    expect(lines).toContain('factors[f0].params.eta[1]: 1 -> 2')
    expect(lines.some((l) => l.startsWith('factors[f1]: removed'))).toBe(true)
  })
})
