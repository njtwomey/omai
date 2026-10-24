import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { distinctTensor, RUNS } from '../../testing/arbitrary.ts'
import { expectConformant } from '../../testing/check.ts'
import { arbitraryGraph } from '../../testing/graphs.ts'
import { expectKernelLaws, expectPsd, points } from '../../testing/kernels.ts'
import { cycleGraph, graph, laplacian, pathGraph } from '../graph/index.ts'
import { expm } from '../numerics/linalg/index.ts'
import { add, eye, matmul, mul, tensor, toArray, toFlat, type Value } from '../tensor/index.ts'
import {
  additiveChi2Kernel,
  chi2Distance,
  chi2Kernel,
  constantKernel,
  diffusionKernel,
  histogramIntersectionKernel,
  jaccardKernel,
  linearKernel,
  rbfKernel,
  regularisedLaplacianKernel,
  setKernel,
  skewedChi2Kernel,
  spectrumKernel,
  subsequenceKernel,
  theta,
  whiteNoiseKernel,
  withTheta,
  wlSubtreeFeatures,
  wlSubtreeKernel,
} from './index.ts'

/** The subsequence kernel by enumerating index tuples (its definition), for small strings. */
function bruteSsk(s: string, t: string, n: number, lambda: number): number {
  const tuples = (len: number): number[][] => {
    const out: number[][] = []
    const walk = (start: number, acc: number[]): void => {
      if (acc.length === n) {
        out.push(acc)
        return
      }
      for (let i = start; i < len; i++) walk(i + 1, [...acc, i])
    }
    walk(0, [])
    return out
  }
  let total = 0
  for (const i of tuples(s.length)) {
    const u = i.map((p) => s[p]).join('')
    for (const j of tuples(t.length)) {
      if (j.map((q) => t[q]).join('') !== u) continue
      const span = (ix: number[]): number => (ix[ix.length - 1] ?? 0) - (ix[0] ?? 0) + 1
      total += lambda ** (span(i) + span(j))
    }
  }
  return total
}

const words = fc.array(fc.string({ unit: fc.constantFrom('a', 'b', 'c'), minLength: 0, maxLength: 7 }), {
  minLength: 2,
  maxLength: 6,
})

const X = tensor([
  [0.4, 1.2, 0.3],
  [1.1, 0.25, 0.9],
  [0.6, 0.8, 1.7],
])
const Y = tensor([
  [0.5, 0.35, 1.3],
  [1.4, 0.6, 0.2],
])

describe('white noise and constant kernels', () => {
  it('white noise is sigma^2 I on a set and zero across sets', () => {
    const k = whiteNoiseKernel({ noiseLevel: 0.3 })
    expect(toArray(k.k([[1], [2]]))).toEqual([
      [0.3, 0],
      [0, 0.3],
    ])
    expect(toArray(k.k([[1], [2]], [[1]]))).toEqual([[0], [0]])
    expect(toArray(k.diag([[1], [2]]))).toEqual([0.3, 0.3])
    fc.assert(
      fc.property(points(5, 2), points(2, 2), (x, y) => {
        expectKernelLaws(k, x, y, { crossIsGram: false })
      }),
      { numRuns: RUNS },
    )
  })
  it('the constant kernel is constant, PSD and differentiable in its value', () => {
    const k = constantKernel({ value: 1.7 })
    fc.assert(
      fc.property(points(5, 2), points(2, 2), (x, y) => {
        expectKernelLaws(k, x, y)
      }),
      { numRuns: RUNS },
    )
    expectConformant((t: Value) => withTheta(k, t).k(X, Y), [theta(k)])
  })
})

describe('histogram kernels', () => {
  const kernels = [
    ['chi2', chi2Kernel({ gamma: 0.6 })],
    ['additive chi2', additiveChi2Kernel()],
    ['skewed chi2', skewedChi2Kernel({ skewedness: 0.4 })],
    ['histogram intersection', histogramIntersectionKernel()],
  ] as const
  for (const [name, k] of kernels) {
    it(`${name}: PSD on non-negative features`, () => {
      fc.assert(
        fc.property(points(6, 4, 0, 2), points(3, 4, 0, 2), (x, y) => {
          expectKernelLaws(k, x, y)
        }),
        { numRuns: RUNS },
      )
    })
    it(`${name}: differentiable in the features and in theta`, () => {
      expectConformant((x: Value) => k.k(x, Y), [X])
      if ((theta(k).shape[0] ?? 0) > 0) expectConformant((t: Value) => withTheta(k, t).k(X, Y), [theta(k)])
    })
  }
  it('histogram intersection is differentiable away from ties', () => {
    fc.assert(
      fc.property(distinctTensor([2, 3], 0.1, 2), (x) => {
        expectConformant((v: Value) => histogramIntersectionKernel().k(v, [[0.05, 5, 0.07]]), [x])
      }),
      { numRuns: 5 },
    )
  })
  it('the chi-square distance is zero on the diagonal and treats 0 / 0 as 0', () => {
    expect(
      toArray(
        chi2Distance([
          [0, 1],
          [0, 3],
        ]),
      ),
    ).toEqual([
      [0, 1],
      [1, 0],
    ])
  })
})

describe('string kernels', () => {
  it('the spectrum kernel counts shared k-mers (brute force), is PSD and normalises to a unit diagonal', () => {
    fc.assert(
      fc.property(words, fc.integer({ min: 1, max: 3 }), (ws, k) => {
        const counts = (s: string): Map<string, number> => {
          const m = new Map<string, number>()
          for (let i = 0; i + k <= s.length; i++) m.set(s.slice(i, i + k), (m.get(s.slice(i, i + k)) ?? 0) + 1)
          return m
        }
        const got = toFlat(spectrumKernel(ws, undefined, { k }))
        ws.forEach((s, i) => {
          ws.forEach((t, j) => {
            let want = 0
            for (const [w, c] of counts(s)) want += c * (counts(t).get(w) ?? 0)
            expect(got[i * ws.length + j]).toBe(want)
          })
        })
        expectPsd(spectrumKernel(ws, undefined, { k }))
        const norm = toFlat(spectrumKernel(ws, undefined, { k, normalise: true }))
        ws.forEach((s, i) => {
          expect(norm[i * ws.length + i]).toBeCloseTo(s.length >= k ? 1 : 0, 12)
        })
      }),
      { numRuns: RUNS },
    )
  })
  it('the subsequence kernel matches its definition and is PSD', () => {
    fc.assert(
      fc.property(words, fc.integer({ min: 1, max: 3 }), fc.double({ min: 0.1, max: 1, noNaN: true }), (ws, n, lam) => {
        const got = toFlat(subsequenceKernel(ws, undefined, { n, lambda: lam, normalise: false }))
        ws.forEach((s, i) => {
          ws.forEach((t, j) => {
            const want = bruteSsk(s, t, n, lam)
            expect(Math.abs((got[i * ws.length + j] ?? 0) - want)).toBeLessThanOrEqual(1e-12 * Math.max(1, want))
          })
        })
        expectPsd(subsequenceKernel(ws, undefined, { n, lambda: lam }), 1e-8)
      }),
      { numRuns: RUNS },
    )
  })
  it('rejects invalid settings', () => {
    expect(() => spectrumKernel(['a'], undefined, { k: 0 })).toThrow()
    expect(() => subsequenceKernel(['a'], undefined, { n: 0 })).toThrow()
    expect(() => subsequenceKernel(['a'], undefined, { lambda: 1.5 })).toThrow()
  })
})

describe('set kernels', () => {
  const sets = fc.array(points(3, 2), { minLength: 2, maxLength: 5 })
  it('the mean-map kernel averages the base kernel and is PSD', () => {
    fc.assert(
      fc.property(sets, (ss) => {
        const k = rbfKernel({ lengthScale: 0.8 })
        const m = setKernel(k, ss)
        expectPsd(m)
        const first = toFlat(k.k(ss[0] as Value, ss[1] as Value))
        expect(toFlat(m)[1]).toBeCloseTo(first.reduce((a, b) => a + b, 0) / first.length, 12)
        const norm = toFlat(setKernel(k, ss, undefined, { normalise: true }))
        ss.forEach((_, i) => {
          expect(norm[i * ss.length + i]).toBeCloseTo(1, 12)
        })
      }),
      { numRuns: RUNS },
    )
  })
  it('the sum variant is differentiable in the points of a set', () => {
    const other = [
      tensor([[0.2, 0.1]]),
      tensor([
        [1, -1],
        [0.5, 0.5],
      ]),
    ]
    expectConformant(
      (s: Value) => setKernel(rbfKernel(), [s, [[0.3, 0.3]]], other, { reduce: 'sum' }),
      [
        tensor([
          [0.1, 0.4],
          [0.9, -0.2],
        ]),
      ],
    )
    expect(toArray(setKernel(linearKernel(), [[[1], [2]]], [[[3]]], { reduce: 'sum' }))).toEqual([[9]])
  })
  it('the Jaccard kernel is PSD and 1 between empty sets', () => {
    const labelSets = fc.array(fc.uniqueArray(fc.integer({ min: 0, max: 6 }), { maxLength: 5 }), {
      minLength: 2,
      maxLength: 6,
    })
    fc.assert(
      fc.property(labelSets, (ls) => {
        expectPsd(jaccardKernel(ls))
      }),
      { numRuns: RUNS },
    )
    expect(toArray(jaccardKernel([[]], [[], [1]]))).toEqual([[1, 0]])
  })
})

describe('graph kernels', () => {
  it('the diffusion kernel is exp(-beta L), PSD and differentiable in beta', () => {
    fc.assert(
      fc.property(
        arbitraryGraph({ minNodes: 2, maxNodes: 8 }),
        fc.double({ min: 0.05, max: 2, noNaN: true }),
        (g, b) => {
          const k = diffusionKernel(g, b)
          expectPsd(k)
          const want = toFlat(expm(mul(laplacian(g), -b)))
          toFlat(k).forEach((v, i) => {
            expect(Math.abs(v - (want[i] ?? 0))).toBeLessThan(1e-9)
          })
        },
      ),
      { numRuns: RUNS },
    )
    const l = laplacian(cycleGraph(5))
    expectConformant((b: Value) => diffusionKernel(l, b), [tensor(0.7)])
  })
  it('the regularised Laplacian kernel inverts I + sigma^2 L, is PSD and differentiable in sigma^2', () => {
    fc.assert(
      fc.property(arbitraryGraph({ minNodes: 2, maxNodes: 8 }), fc.double({ min: 0, max: 3, noNaN: true }), (g, s) => {
        const k = regularisedLaplacianKernel(g, s)
        expectPsd(k)
        const n = g.n
        const product = toFlat(matmul(add(eye(n), mul(laplacian(g), s)), k))
        toFlat(eye(n)).forEach((v, i) => {
          expect(Math.abs((product[i] ?? 0) - v)).toBeLessThan(1e-9)
        })
      }),
      { numRuns: RUNS },
    )
    expectConformant((s: Value) => regularisedLaplacianKernel(pathGraph(4), s, { normalised: true }), [tensor(0.6)])
  })
  it('the WL subtree kernel is PSD, invariant to relabelling, and counts nodes at round 0', () => {
    fc.assert(
      fc.property(
        fc.array(arbitraryGraph({ minNodes: 1, maxNodes: 7, density: 0.4 }), { minLength: 2, maxLength: 5 }),
        (gs) => {
          expectPsd(wlSubtreeKernel(gs, undefined, { iterations: 3 }))
          const k0 = toFlat(wlSubtreeKernel(gs, undefined, { iterations: 0 }))
          gs.forEach((g, i) => {
            gs.forEach((h, j) => {
              expect(k0[i * gs.length + j]).toBe(g.n * h.n)
            })
          })
          // Reverse the node numbering: an isomorphic graph has the same features.
          const g = gs[0] as (typeof gs)[number]
          const flipped = graph(
            g.edges.map((e) => [g.n - 1 - e.source, g.n - 1 - e.target] as [number, number]),
            { nodes: g.n },
          )
          expect(toArray(wlSubtreeKernel([flipped], gs))).toEqual(toArray(wlSubtreeKernel([g], gs)))
          const norm = toFlat(wlSubtreeKernel(gs, undefined, { normalise: true }))
          gs.forEach((_, i) => {
            expect(norm[i * gs.length + i]).toBeCloseTo(1, 12)
          })
        },
      ),
      { numRuns: RUNS },
    )
  })
  it('WL features separate a path from a star, and use node labels', () => {
    const path = pathGraph(4)
    const star = graph([
      [0, 1],
      [0, 2],
      [0, 3],
    ])
    const k = toFlat(wlSubtreeKernel([path, star], undefined, { iterations: 2, normalise: true }))
    expect(k[1]).toBeLessThan(1)
    const labelled = wlSubtreeFeatures(path, { iterations: 0, nodeLabels: [['x', 'y', 'y', 'x']] })
    expect([...labelled.entries()].sort()).toEqual([
      ['0:x', 2],
      ['0:y', 2],
    ])
    expect(() => wlSubtreeFeatures(path, { iterations: -1 })).toThrow()
    // Distinct initial labels for the second set of graphs.
    const other = toFlat(
      wlSubtreeKernel([path], [path], {
        iterations: 0,
        nodeLabels: [['a', 'a', 'a', 'a']],
        otherNodeLabels: [['a', 'b', 'b', 'a']],
      }),
    )
    expect(other).toEqual([8])
  })
})
