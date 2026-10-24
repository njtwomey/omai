import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { consult, prove } from '../index.ts'
import { foil, foilGain, type Tuple } from './index.ts'

const NODES = ['n0', 'n1', 'n2', 'n3', 'n4']
const graph = fc.uniqueArray(fc.tuple(fc.constantFrom(...NODES), fc.constantFrom(...NODES)), {
  minLength: 2,
  maxLength: 8,
  selector: ([a, b]) => `${a}${b}`,
})

describe('FOIL properties', () => {
  it('a complete theory proves every positive and no negative', () => {
    fc.assert(
      fc.property(graph, fc.constantFrom('two-step', 'symmetric'), (edges, kind) => {
        const pairs = new Set(edges.map(([a, b]) => `${a},${b}`))
        const positive: Tuple[] = []
        for (const x of NODES) {
          for (const y of NODES) {
            const yes =
              kind === 'two-step'
                ? NODES.some((z) => pairs.has(`${x},${z}`) && pairs.has(`${z},${y}`))
                : pairs.has(`${x},${y}`) && pairs.has(`${y},${x}`)
            if (yes) positive.push([x, y])
          }
        }
        fc.pre(positive.length > 0)
        const r = foil({ target: 't', arity: 2, positive, background: { e: edges } })
        if (r.status !== 'complete') return
        const db = consult(edges.map(([a, b]) => `e(${a}, ${b}).`).join(' '))
        const theory = consult(r.clauses, { base: db })
        // The closed world is over the constants the task mentions.
        const seen = NODES.filter((n) => edges.some(([a, b]) => a === n || b === n))
        for (const x of seen) {
          for (const y of seen) {
            const expected = positive.some(([a, b]) => a === x && b === y)
            expect(prove(theory, `t(${x}, ${y})`, { unknown: 'fail' })).toBe(expected)
          }
        }
      }),
      { numRuns: 80 },
    )
  })

  it('gain is zero without positive tuples and grows with purity', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 20 }),
        fc.integer({ min: 0, max: 20 }),
        fc.integer({ min: 1, max: 20 }),
        (p, n, t) => {
          expect(foilGain(p, n, 0, 5, 0)).toBe(0)
          expect(foilGain(p, n, p, 0, Math.min(t, p))).toBeGreaterThanOrEqual(0)
          if (n > 0) expect(foilGain(p, n, p, 0, p)).toBeGreaterThan(foilGain(p, n, p, n, p))
        },
      ),
    )
  })
})
