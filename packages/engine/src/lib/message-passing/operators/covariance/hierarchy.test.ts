// The joint site of a group of vectors sharing a mean and a Wishart precision (`mvn-wishart-group`) against a brute-force reference that
// does not use its algebra: a grid over the three entries of the precision matrix, and at each node the joint Gaussian of (mean, vectors)
// with the cavities read as noisy readings of the vectors (Gaussian conditioning on a dense joint covariance).
import { describe, expect, it } from 'vitest'
import { importanceOfGroup, type Reading } from '../../../../testing/covariance-group.ts'
import { nodesOn } from '../../../../testing/message-passing.ts'
import { messageArg, valueArg } from '../../core/args.ts'
import { wishart, wishartEta, wishartParams } from '../../core/families-matrix.ts'
import { gaussianMv, gaussianMvEta } from '../../core/families-vector.ts'
import type { Arg, Eta, Memo, OperatorContext } from '../../core/types.ts'
import { spd } from '../../numerics/_dense.ts'
import { mvnWishart } from './covariance.ts'
import { mvnWishartGroup } from './hierarchy.ts'

const W = wishart(2)
const G = gaussianMv(2)
const G_OF = gaussianMv

const ctx = (
  to: number,
  args: readonly Arg[],
  groups: number,
  mode: 'ep' | 'vmp' = 'ep',
  memo?: Memo,
): OperatorContext => ({
  mode,
  to,
  args,
  params: { groups },
  power: 1,
  ...(memo === undefined ? {} : { memo }),
})

const inverse2 = (a: number, b: number, c: number): [number, number, number] => {
  const det = a * c - b * b
  return [c / det, -b / det, a / det]
}

/**
 * The tilted distribution of the precision matrix on a midpoint grid in (L11, L12, L22) and the moments of each Gaussian under it. At a node the
 * vectors and the mean are one Gaussian with the mean's cavity as prior, each vector `N(mean, L^-1)` around it, and the cavity of vector `j` is a reading
 * `N(m_j, V_j)` of it: the evidence is the density of the readings, the conditionals the Gaussian conditioning on them.
 */
function brute(
  cavity: Eta,
  vectors: readonly Reading[],
  mean: Reading,
  panels = 1,
): {
  readonly logZ: number
  readonly logDet: number
  readonly lambda: number[]
  readonly mean: number[][]
  readonly cov: number[][]
} {
  const count = vectors.length
  // a box around the cavity wide enough to hold the tilted matrix (a Gauss-Legendre rule of eight points on each of `panels` panels per axis)
  const p = wishartParams(cavity, 2) as NonNullable<ReturnType<typeof wishartParams>>
  const sd = (i: number, j: number): number =>
    Math.sqrt(
      p.nu * ((p.scale[i * 2 + j] as number) ** 2 + (p.scale[i * 2 + i] as number) * (p.scale[j * 2 + j] as number)),
    )
  // the diagonal on panels that thicken toward zero (a small precision is where the group effects are least pooled), the correlation r = L12 / sqrt(L11 L22) on (-1, 1)
  const graded = (hi: number): { x: number[]; w: number[] } => {
    const edges = [1e-7, 1e-3, 1e-2, 0.05, 0.15, 0.4, 1, 2.5, 5, 10, 20, Math.max(40, hi)]
    const out = { x: [] as number[], w: [] as number[] }
    for (let e = 0; e + 1 < edges.length; e++) {
      const part = nodesOn(edges[e] as number, edges[e + 1] as number, panels, 8)
      out.x.push(...part.x)
      out.w.push(...part.w)
    }
    return out
  }
  const axes = [
    graded((p.mean[0] as number) + 14 * sd(0, 0)),
    nodesOn(-1, 1, panels * 2, 8),
    graded((p.mean[3] as number) + 14 * sd(1, 1)),
  ] as const
  const size = 2 * (count + 1)
  const nodes: { l: [number, number, number]; w: number; mean: number[][]; cov: number[][] }[] = []
  let total = 0
  for (let i = 0; i < axes[0].x.length; i++)
    for (let j = 0; j < axes[1].x.length; j++)
      for (let k = 0; k < axes[2].x.length; k++) {
        const l11 = axes[0].x[i] as number
        const l22 = axes[2].x[k] as number
        const l12 = (axes[1].x[j] as number) * Math.sqrt(l11 * l22)
        const det = l11 * l22 - l12 * l12
        if (det <= 0) continue
        const lq = W.pair(cavity, [Math.log(det), l11, l12, l12, l22]) - W.logPartition(cavity)
        const [s11, s12, s22] = inverse2(l11, l12, l22)
        // the joint prior of z = (mean, x_1..x_J): Cov(mean) = V0, Cov(x_j, mean) = V0, Cov(x_j, x_k) = V0 + delta L^-1
        const sigma = new Float64Array(size * size)
        const block = (a: number, b: number, extra: readonly number[] | undefined): void => {
          for (let r = 0; r < 2; r++)
            for (let c = 0; c < 2; c++)
              sigma[(2 * a + r) * size + 2 * b + c] =
                (mean.v[r * 2 + c] as number) + (extra === undefined ? 0 : (extra[r * 2 + c] as number))
        }
        for (let a = 0; a <= count; a++)
          for (let b = 0; b <= count; b++) block(a, b, a === b && a > 0 ? [s11, s12, s12, s22] : undefined)
        const z0 = Array.from({ length: size }, (_, e) => mean.m[e % 2] as number)
        // readings y_j = x_j + noise V_j: S = H sigma H' + R, the cross covariance sigma H'
        const q = 2 * count
        const s = new Float64Array(q * q)
        const cross = new Float64Array(size * q)
        for (let a = 0; a < count; a++)
          for (let b = 0; b < count; b++)
            for (let r = 0; r < 2; r++)
              for (let c = 0; c < 2; c++)
                s[(2 * a + r) * q + 2 * b + c] =
                  (sigma[(2 * (a + 1) + r) * size + 2 * (b + 1) + c] as number) +
                  (a === b ? ((vectors[a] as Reading).v[r * 2 + c] as number) : 0)
        for (let e = 0; e < size; e++)
          for (let b = 0; b < count; b++)
            for (let c = 0; c < 2; c++) cross[e * q + 2 * b + c] = sigma[e * size + 2 * (b + 1) + c] as number
        const f = spd(s, q) as NonNullable<ReturnType<typeof spd>>
        const resid = vectors.flatMap((v, a) => [0, 1].map((r) => (v.m[r] as number) - (z0[2 * (a + 1) + r] as number)))
        const alpha = f.solve(resid)
        let quad = 0
        for (let e = 0; e < q; e++) quad += (resid[e] as number) * (alpha[e] as number)
        // the unnormalised cavities carry their own masses: N(x; m, V) -> exp(h x - x'Wx/2) multiplies by (2 pi) sqrt|V| exp(m'Wm/2)
        const logDensity = -0.5 * (q * Math.log(2 * Math.PI) + f.logDet + quad)
        const w =
          Math.exp(lq + logDensity) *
          (axes[0].w[i] as number) *
          (axes[1].w[j] as number) *
          (axes[2].w[k] as number) *
          Math.sqrt(l11 * l22)
        const sInv = f.inverse()
        const mu: number[][] = []
        const cv: number[][] = []
        for (let a = 0; a <= count; a++) {
          const m2 = [0, 1].map((r) => {
            let acc = z0[2 * a + r] as number
            for (let c = 0; c < q; c++) acc += (cross[(2 * a + r) * q + c] as number) * (alpha[c] as number)
            return acc
          })
          const c2: number[] = []
          for (let r = 0; r < 2; r++)
            for (let c = 0; c < 2; c++) {
              let acc = sigma[(2 * a + r) * size + 2 * a + c] as number
              for (let e = 0; e < q; e++)
                for (let g = 0; g < q; g++)
                  acc -=
                    (cross[(2 * a + r) * q + e] as number) *
                    (sInv[e * q + g] as number) *
                    (cross[(2 * a + c) * q + g] as number)
              c2.push(acc)
            }
          mu.push(m2)
          cv.push(c2)
        }
        nodes.push({ l: [l11, l12, l22], w, mean: mu, cov: cv })
        total += w
      }
  const expect1 = (f: (n2: (typeof nodes)[number]) => number): number =>
    nodes.reduce((acc, n2) => acc + n2.w * f(n2), 0) / total
  // the ports as the operator has them: the vectors, then the mean (the joint has the mean first)
  const ports = [...Array.from({ length: count }, (_, a) => a + 1), 0]
  const meanOf = ports.map((a) => [0, 1].map((r) => expect1((n2) => (n2.mean[a] as number[])[r] as number)))
  const covOf = ports.map((a, at) =>
    [0, 1, 2, 3].map((e) => {
      const r = Math.floor(e / 2)
      const c = e % 2
      return (
        expect1(
          (n2) =>
            ((n2.cov[a] as number[])[e] as number) +
            ((n2.mean[a] as number[])[r] as number) * ((n2.mean[a] as number[])[c] as number),
        ) -
        ((meanOf[at] as number[])[r] as number) * ((meanOf[at] as number[])[c] as number)
      )
    }),
  )
  return {
    // the masses of the unnormalised cavities (each reading as exp(h x - x' W x / 2)) are added by the caller
    logZ: Math.log(total) + W.logPartition(cavity),
    logDet: expect1((n2) => Math.log(n2.l[0] * n2.l[2] - n2.l[1] ** 2)),
    lambda: [expect1((n2) => n2.l[0]), expect1((n2) => n2.l[1]), expect1((n2) => n2.l[2])],
    mean: meanOf,
    cov: covOf,
  }
}

/** log of the mass of `exp(h x - x' W x / 2)` for the Gaussian of mean `m` and covariance `v` (its unnormalised cavity). */
const cavityMass = (m: readonly number[], v: readonly number[]): number => {
  const e = gaussianMvEta(m, v)
  return G.logPartition(e)
}

const vectors: Reading[] = [
  { m: [0.9, -0.4], v: [0.08, 0.01, 0.01, 0.05] },
  { m: [-0.6, 0.5], v: [0.06, -0.01, -0.01, 0.07] },
  { m: [0.2, -1.1], v: [0.1, 0.02, 0.02, 0.06] },
]
const populationCavity: Reading = { m: [0.1, 0.0], v: [4, 0, 0, 4] }
const matrixCavity = wishartEta(4, [0.5, 0.1, 0.1, 0.4], 2)

const argsOf = (vs: readonly Reading[], mean: Reading, cavity: Eta): Arg[] => [
  ...vs.map((r) => messageArg(G, gaussianMvEta(r.m, r.v))),
  messageArg(G, gaussianMvEta(mean.m, mean.v)),
  messageArg(W, cavity),
]

describe('mvn-wishart-group: one site for a group sharing a mean and a precision matrix', () => {
  const args = argsOf(vectors, populationCavity, matrixCavity)
  const count = vectors.length
  const ref = brute(matrixCavity, vectors, populationCavity)

  it('has a port per vector, the mean and the matrix, and takes the vectors as variables or constants', () => {
    expect(mvnWishartGroup.ports).toBeTypeOf('function')
    const ports = (mvnWishartGroup.ports as (p: { groups: number }) => readonly { name: string }[])({ groups: 3 })
    expect(ports.map((q) => q.name)).toEqual(['x0', 'x1', 'x2', 'mean', 'precision'])
    expect(mvnWishartGroup.check?.({ groups: 3 }, [G, G, G, G, W])).toBeUndefined()
    expect(mvnWishartGroup.check?.({ groups: 3 }, [G, G, G, W])).toMatch(/one port each/)
  })

  it('the message to the matrix matches the brute-force tilted moments, and the evidence the integral', () => {
    const msg = mvnWishartGroup.ep?.(ctx(count + 1, args, count)).message as number[]
    const matched = W.moments(W.product(matrixCavity, msg))
    expect(Math.abs((matched[0] as number) - ref.logDet), 'E log det').toBeLessThan(4e-3)
    expect(matched[1], 'E L11').toBeCloseTo(ref.lambda[0] as number, 1)
    expect((matched[1] as number) / (ref.lambda[0] as number)).toBeCloseTo(1, 2)
    expect(matched[2], 'E L12').toBeCloseTo(ref.lambda[1] as number, 1)
    expect((matched[4] as number) / (ref.lambda[2] as number)).toBeCloseTo(1, 2)
    const masses = [...vectors, populationCavity].reduce((acc, r) => acc + cavityMass(r.m, r.v), 0)
    expect(mvnWishartGroup.logEvidence?.(ctx(count + 1, args, count))).toBeCloseTo(ref.logZ + masses, 2)
  })

  it('the message to each vector and to the mean is the moment match of the mixture of its conditionals', () => {
    for (let port = 0; port <= count; port++) {
      const cavityOf = port < count ? vectors[port] : populationCavity
      const msg = mvnWishartGroup.ep?.(ctx(port, args, count)).message as number[]
      const posterior = (args[port] as Arg).family.product((args[port] as Arg).eta as Eta, msg)
      const mean = (G.moments(posterior) as number[]).slice(0, 2)
      const second = (G.moments(posterior) as number[]).slice(2)
      const cov = second.map((v, e) => v - (mean[Math.floor(e / 2)] as number) * (mean[e % 2] as number))
      void cavityOf
      expect((mean[0] as number) - ((ref.mean[port] as number[])[0] as number), `mean ${port}`).toBeCloseTo(0, 2)
      expect((mean[1] as number) - ((ref.mean[port] as number[])[1] as number), `mean ${port}`).toBeCloseTo(0, 2)
      for (const e of [0, 3])
        expect((cov[e] as number) / ((ref.cov[port] as number[])[e] as number), `variance ${port}.${e}`).toBeCloseTo(
          1,
          2,
        )
      expect(cov[1] as number, `covariance ${port}`).toBeCloseTo((ref.cov[port] as number[])[1] as number, 2)
    }
  })

  it.each([2, 3, 4])(
    'in the limits of exactly known vectors it is the Wishart conjugate update, in dimension %d',
    (d) => {
      const Wd = wishart(d)
      const rows = [
        [0.9, -0.4, 0.3, 0.5],
        [-0.6, 0.5, 0.1, -0.2],
        [0.2, -1.1, -0.5, 0.4],
        [0.5, 0.2, 0.8, -0.7],
        [-0.2, 0.7, -0.3, 0.1],
      ].map((r) => r.slice(0, d))
      const identity = (v: number): number[] =>
        Array.from({ length: d * d }, (_, e) => (e % d === Math.floor(e / d) ? v : 0))
      const prior = wishartEta(d + 2, identity(0.5), d)
      const exactVectors = rows.map((r) => messageArg(gaussianMv(d), gaussianMvEta(r, identity(1e-9))))
      // a mean that is known (a tight cavity): the rows about it, all of their df; a mean that is flat: the rows about their own mean, one df fewer
      for (const [variance, lost] of [
        [1e-9, 0],
        [1e9, 1],
      ] as const) {
        const centre = Array.from({ length: d }, (_, k) =>
          lost === 0 ? 0.1 : rows.reduce((acc, r) => acc + (r[k] as number), 0) / rows.length,
        )
        const args = [
          ...exactVectors,
          messageArg(gaussianMv(d), gaussianMvEta(Array(d).fill(0.1), identity(variance))),
          messageArg(Wd, prior),
        ]
        const message = mvnWishartGroup.ep?.({ ...ctx(rows.length + 1, args, rows.length) }).message as number[]
        const scatter = new Array<number>(d * d).fill(0)
        for (const r of rows)
          for (let a = 0; a < d; a++)
            for (let b = 0; b < d; b++)
              scatter[a * d + b] =
                (scatter[a * d + b] as number) +
                ((r[a] as number) - (centre[a] as number)) * ((r[b] as number) - (centre[b] as number))
        const exact = Wd.moments(Wd.product(prior, [0.5 * (rows.length - lost), ...scatter.map((v) => -0.5 * v)]))
        const got = Wd.moments(Wd.product(prior, message))
        // a logarithm is not a polynomial: the five nodes of dimension three carry its error (about 0.013)
        expect(got[0] as number, 'E log det').toBeCloseTo(exact[0] as number, d === 2 ? 2 : 1)
        for (let e = 1; e < got.length; e++)
          expect(got[e] as number, `E L[${e - 1}] with variance ${variance}`).toBeCloseTo(
            exact[e] as number,
            d === 4 ? 1 : 2,
          )
      }
    },
  )

  it.each([3, 4])(
    'in dimension %d the matrix and the mean match an importance-sampling reference',
    (d) => {
      const Wd = wishart(d)
      const identity = (v: number): number[] =>
        Array.from({ length: d * d }, (_, e) => (e % d === Math.floor(e / d) ? v : 0))
      const base = [
        [0.9, -0.4, 0.3, 0.5],
        [-0.6, 0.5, 0.1, -0.2],
        [0.2, -1.1, -0.5, 0.4],
        [0.5, 0.2, 0.8, -0.7],
      ]
      const vs: Reading[] = base.map((r, j) => ({ m: r.slice(0, d), v: identity(0.05 + 0.02 * j) }))
      const population: Reading = { m: Array(d).fill(0.1) as number[], v: identity(4) }
      const cavity = wishartEta(d + 2, identity(0.5), d)
      const g = gaussianMv(d)
      const args = [
        ...vs.map((r) => messageArg(g, gaussianMvEta(r.m, r.v))),
        messageArg(g, gaussianMvEta(population.m, population.v)),
        messageArg(Wd, cavity),
      ]
      const reference = importanceOfGroup(d, vs, population, cavity, 6000)
      expect(reference.ess).toBeGreaterThan(1500)
      const message = mvnWishartGroup.ep?.(ctx(vs.length + 1, args, vs.length)).message as number[]
      const matched = Wd.moments(Wd.product(cavity, message))
      // E L entry by entry, against the sampling error of the reference (about 3%)
      for (let e = 0; e < d * d; e++)
        expect(matched[1 + e] as number, `E L[${e}]`).toBeCloseTo(reference.lambda[e] as number, 0)
      for (const i of Array.from({ length: d }, (_, k) => k * d + k))
        expect((matched[1 + i] as number) / (reference.lambda[i] as number), `E L[${i}] relative`).toBeCloseTo(1, 1)
      const meanMessage = mvnWishartGroup.ep?.(ctx(vs.length, args, vs.length)).message as number[]
      const posterior = G_OF(d).product(gaussianMvEta(population.m, population.v), meanMessage)
      const mo = G_OF(d).moments(posterior)
      for (let i = 0; i < d; i++) {
        expect(mo[i] as number, `mean of the mean ${i}`).toBeCloseTo(reference.meanOfMean[i] as number, 1)
        const variance = (mo[d + i * d + i] as number) - (mo[i] as number) ** 2
        expect(variance / (reference.meanCov[i * d + i] as number), `variance of the mean ${i}`).toBeCloseTo(1, 1)
      }
    },
    60_000,
  )

  it('shares one tilt between its ports and its evidence when the engine gives it a cache, and gives the values of no cache', () => {
    const memo: Memo = new Map()
    const cached = Array.from(
      { length: count + 2 },
      (_, to) => mvnWishartGroup.ep?.(ctx(to, args, count, 'ep', memo)).message,
    )
    mvnWishartGroup.logEvidence?.(ctx(0, args, count, 'ep', memo))
    expect(memo.size).toBe(1)
    const plain = Array.from({ length: count + 2 }, (_, to) => mvnWishartGroup.ep?.(ctx(to, args, count)).message)
    expect(cached).toEqual(plain)
  })

  it('with one vector it is the per-vector factor: the same messages to the matrix, the vector and the mean', () => {
    const one = [args[0] as Arg, args[count] as Arg, args[count + 1] as Arg]
    const single = [0, 1, 2].map((to) => mvnWishart.ep?.({ ...ctx(to, one, 1), params: {} }).message as number[])
    const group = [0, 1, 2].map((to) => mvnWishartGroup.ep?.(ctx(to, one, 1)).message as number[])
    for (let to = 0; to < 3; to++)
      for (let e = 0; e < (single[to] as number[]).length; e++)
        expect((group[to] as number[])[e], `port ${to}, entry ${e}`).toBeCloseTo(
          (single[to] as number[])[e] as number,
          1,
        )
  })

  it('with one vector in dimension 4 it is the per-vector factor (which has the same three-node rule there)', () => {
    const d = 4
    const identity = (v: number): number[] =>
      Array.from({ length: d * d }, (_, e) => (e % d === Math.floor(e / d) ? v : 0))
    const g = gaussianMv(d)
    const one = [
      messageArg(g, gaussianMvEta([0.9, -0.4, 0.3, 0.5], identity(0.1))),
      messageArg(g, gaussianMvEta([0.1, 0.1, 0.1, 0.1], identity(2))),
      messageArg(wishart(d), wishartEta(d + 2, identity(0.5), d)),
    ]
    const single = mvnWishart.ep?.({ ...ctx(2, one, 1), params: {} }).message as number[]
    const group = mvnWishartGroup.ep?.(ctx(2, one, 1)).message as number[]
    for (let e = 0; e < single.length; e++) expect(group[e] as number, `entry ${e}`).toBeCloseTo(single[e] as number, 1)
  }, 60_000)

  it('reads a vector that says nothing as absent, and a constant vector as an exact reading', () => {
    // a flat vector does not change the matrix's message: the others are told as if it were not there
    const flat = [...args.slice(0, count), messageArg(G, G.flat), ...args.slice(count)]
    const without = mvnWishartGroup.ep?.(ctx(count + 1, args, count)).message as number[]
    const withFlat = mvnWishartGroup.ep?.(ctx(count + 2, flat, count + 1)).message as number[]
    expect(withFlat.every(Number.isFinite)).toBe(true)
    expect(withFlat.length).toBe(without.length)
    // a constant vector is a reading with no variance: close to a very narrow cavity, and a message to it is not asked for
    const narrow = [messageArg(G, gaussianMvEta([0.9, -0.4], [1e-8, 0, 0, 1e-8])), ...args.slice(1)]
    const exact = [valueArg(G, [0.9, -0.4]), ...args.slice(1)]
    const a = mvnWishartGroup.ep?.(ctx(count + 1, narrow, count)).message as number[]
    const b = mvnWishartGroup.ep?.(ctx(count + 1, exact, count)).message as number[]
    for (let e = 0; e < a.length; e++) expect(b[e] as number).toBeCloseTo(a[e] as number, 2)
  })

  it('the variational rule is the sum of the per-vector rules, and its bound is the sum of theirs', () => {
    const q = [
      messageArg(G, gaussianMvEta([0.9, -0.4], [0.08, 0, 0, 0.05])),
      messageArg(G, gaussianMvEta([-0.6, 0.5], [0.06, 0, 0, 0.07])),
      messageArg(G, gaussianMvEta([0.1, 0], [0.5, 0, 0, 0.5])),
      messageArg(W, matrixCavity),
    ]
    const mu = q[2] as Arg
    const lambda = q[3] as Arg
    const pair = (j: number, to: number): number[] =>
      mvnWishart.vmp?.({ ...ctx(to, [q[j] as Arg, mu, lambda], 1, 'vmp'), params: {} }).message as number[]
    const g = (to: number): number[] => mvnWishartGroup.vmp?.(ctx(to, q, 2, 'vmp')).message as number[]
    expect(g(0)).toEqual(pair(0, 0))
    expect(g(1)).toEqual(pair(1, 0))
    expect(g(2)).toEqual(pair(0, 1).map((v, e) => v + (pair(1, 1)[e] as number)))
    expect(g(3)).toEqual(pair(0, 2).map((v, e) => v + (pair(1, 2)[e] as number)))
    const bound = (j: number): number =>
      mvnWishart.expectedLog?.({ ...ctx(0, [q[j] as Arg, mu, lambda], 1, 'vmp'), params: {} }) as number
    expect(mvnWishartGroup.expectedLog?.(ctx(0, q, 2, 'vmp'))).toBeCloseTo(bound(0) + bound(1), 12)
  })
})
