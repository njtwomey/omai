// The hybrid plan: a user-written assignment of blocks of sites to kernels. A sweep of the composite leaves a known posterior
// invariant (one sweep from exact draws), a conjugate Gibbs block with a NUTS block agrees with plain NUTS and with the grid,
// discrete and continuous sites move in one plan (HMC within Gibbs, with and without summing the discrete site out), and the run
// cancels within one sweep, resumes from its serialised state and publishes anytime snapshots.
import { describe, expect, it } from 'vitest'
import { expectCancelAndContinue, expectResumeEqual, expectSnapshotCadence } from '../../../testing/methods.ts'
import { DISCRETE } from '../../../testing/probprog.ts'
import type { Json } from '../../base/index.ts'
import { gamma, normal } from '../../dists/index.ts'
import { normal as drawNormal, foldIn, key } from '../../random/index.ts'
import { sliceSampler } from '../../sampling/index.ts'
import { type Tensor, tensor, toFlat } from '../../tensor/index.ts'
import { session, stateFromJson, stateToJson } from '../../trace/index.ts'
import { type Model, observe, plate, sample } from '../primitives.ts'
import { agree, agreementWithin, fromPosterior, gridPosterior } from '../validation/index.ts'
import {
  blockKernels,
  type HybridBlock,
  type HybridBlockSlot,
  type HybridChainState,
  type HybridRunState,
  hybridMethod,
} from './hybrid.ts'
import { nutsMethod } from './nuts-method.ts'

const Y5 = [0.3, -0.8, 1.4, 0.2, 0.9]
const normalGamma: Model = () => {
  const mu = sample('mu', normal(0, 3))
  const sigma = sample('sigma', gamma(3, 2))
  plate('n', 5, () => observe('y', normal(mu, sigma), Y5))
}

// a ~ N(0, 1), b | a ~ N(a, 1), y_i | b ~ N(b, 0.5): the posterior is Gaussian and known in closed form.
const YS = [0.8, 1.3, 0.4, 1.1]
const chain: Model = () => {
  const a = sample('a', normal(0, 1))
  const b = sample('b', normal(a, 1))
  plate('n', 4, () => observe('y', normal(b, 0.5), YS))
}
const exactChain = (() => {
  const precision = 1 / 2 + YS.length / 0.25
  const vb = 1 / precision
  const mb = (YS.reduce((s, v) => s + v, 0) / 0.25) * vb
  return { mb, vb, meanA: mb / 2, sdA: Math.sqrt(vb / 4 + 0.5), meanB: mb, sdB: Math.sqrt(vb) }
})()

const stats = (x: ArrayLike<number>): { mean: number; sd: number } => {
  const a = Array.from(x)
  const mean = a.reduce((s, v) => s + v, 0) / a.length
  return { mean, sd: Math.sqrt(a.reduce((s, v) => s + (v - mean) ** 2, 0) / (a.length - 1)) }
}

describe('a sweep of the composite leaves the posterior invariant', () => {
  const plans: [string, HybridBlock[]][] = [
    [
      'nuts on a, the exact conjugate conditional of b',
      [
        { sites: ['a'], kernel: blockKernels.nuts() },
        { sites: ['b'], kernel: blockKernels.gibbs() },
      ],
    ],
    [
      'hmc on a, slice on b',
      [
        { sites: ['a'], kernel: blockKernels.hmc({ numSteps: 4 }) },
        { sites: ['b'], kernel: blockKernels.slice({ width: 1 }) },
      ],
    ],
    [
      'mala on b, a random walk on a, the other order',
      [
        { sites: ['b'], kernel: blockKernels.mala() },
        { sites: ['a'], kernel: blockKernels.rwm() },
      ],
    ],
    [
      'a kernel the user wrote',
      [
        { sites: ['a'], kernel: blockKernels.custom('my-slice', (t) => sliceSampler(t, { width: 2 })) },
        { sites: ['b'], kernel: blockKernels.gibbs() },
      ],
    ],
  ]
  for (const [label, blocks] of plans)
    it(label, { timeout: 60_000 }, () => {
      const n = 1500
      const m = hybridMethod(chain, { blocks })
      const z = drawNormal(key(1), [n, 2]).data
      const tuned = { stepSize: 0.6, inverseMassMatrix: Float64Array.of(1) }
      const chains: HybridChainState[] = Array.from({ length: n }, (_, i) => {
        // An exact draw of the posterior: b from its marginal, a from its conditional given b.
        const b = exactChain.mb + Math.sqrt(exactChain.vb) * (z[2 * i] ?? 0)
        const a = b / 2 + Math.sqrt(0.5) * (z[2 * i + 1] ?? 0)
        const slots: HybridBlockSlot[] = blocks.map(() => ({
          warm: null,
          tuned,
          accept: Number.NaN,
          stepSize: Number.NaN,
        }))
        return { values: { a: tensor(a), b: tensor(b) }, blocks: slots, logJoint: 0, kept: [] }
      })
      const state: HybridRunState = { stage: 'sample', iteration: 0, chains, failure: null }
      const s = session(m, {
        config: { numChains: n, numWarmup: 0, numDraws: 1 },
        start: { state, t: 1 },
        steps: 3,
        mode: 'eager',
        seed: 4,
      })
      const post = m.posterior?.(s.state, s.config)
      const a = stats(post?.draws?.['a']?.data ?? [])
      const b = stats(post?.draws?.['b']?.data ?? [])
      expect(Math.abs(a.mean - exactChain.meanA)).toBeLessThan((4 * exactChain.sdA) / Math.sqrt(n))
      expect(Math.abs(b.mean - exactChain.meanB)).toBeLessThan((4 * exactChain.sdB) / Math.sqrt(n))
      expect(Math.abs(a.sd - exactChain.sdA)).toBeLessThan((4 * exactChain.sdA) / Math.sqrt(n))
      expect(Math.abs(b.sd - exactChain.sdB)).toBeLessThan((4 * exactChain.sdB) / Math.sqrt(n))
      void foldIn
    })
})

describe('a conjugate block with a NUTS block', () => {
  it('agrees with plain NUTS and with the grid posterior of a normal with unknown scale', { timeout: 120_000 }, () => {
    const grid = gridPosterior(normalGamma, undefined)
    const oracle = { mu: grid.marginal('mu'), sigma: grid.marginal('sigma') }
    const hybrid = hybridMethod(normalGamma, {
      blocks: [
        { sites: ['mu'], kernel: blockKernels.gibbs() },
        { sites: ['sigma'], kernel: blockKernels.nuts() },
      ],
    })
    const hs = session(hybrid, {
      config: { numChains: 3, numWarmup: 300, numDraws: 800 },
      steps: 100_000,
      mode: 'eager',
      seed: 2,
    })
    expect(hs.status).toBe('converged')
    const nuts = nutsMethod(normalGamma)
    const ns = session(nuts, {
      config: { numChains: 3, numWarmup: 300, numDraws: 800 },
      steps: 100_000,
      mode: 'eager',
      seed: 3,
    })
    const table = agree(
      [
        fromPosterior('hybrid', hybrid.posterior?.(hs.state, hs.config) as never),
        fromPosterior('nuts', nuts.posterior?.(ns.state, ns.config) as never),
      ],
      { oracle, quantities: ['mu', 'sigma'], quantilePoints: 400 },
    )
    expect(agreementWithin(table, {}).map((b) => `${b.row.method} ${b.row.quantity}: ${b.failed.join(',')}`)).toEqual(
      [],
    )
    // The tape of every block: the exact block accepts everything, the NUTS block's step size was tuned.
    const post = hybrid.posterior?.(hs.state, hs.config)
    expect(post?.summary().map((r) => r.site)).toEqual(['mu', 'sigma'])
    const first = hs.state.chains[0]
    expect(first?.blocks[0]?.accept).toBe(1)
    expect(first?.blocks[1]?.tuned?.stepSize).toBeGreaterThan(0)
  })
})

describe('discrete and continuous sites in one plan', () => {
  it('HMC within Gibbs on a mixture, with the discrete labels summed out of the continuous block, matches enumeration', {
    timeout: 120_000,
  }, () => {
    const model = DISCRETE['mixture'] as Model
    const grid = gridPosterior(model, undefined)
    const oracle = { 'locs[0]': grid.marginal('locs[0]'), 'locs[1]': grid.marginal('locs[1]') }
    // The second plan moves the labels and the locations in turn, which couples them: it mixes more slowly, so its tolerance is wider.
    const plans: [HybridBlock[], number][] = [
      [
        [
          { sites: ['locs'], kernel: blockKernels.hmc({ numSteps: 6 }), marginalise: ['z'] },
          { sites: ['z'], kernel: blockKernels.gibbs() },
        ],
        0.15,
      ],
      [
        [
          { sites: ['z'], kernel: blockKernels.gibbs() },
          { sites: ['locs'], kernel: blockKernels.nuts() },
        ],
        0.4,
      ],
    ]
    for (const [blocks, meanError] of plans) {
      const m = hybridMethod(model, { blocks })
      const s = session(m, {
        config: { numChains: 3, numWarmup: 150, numDraws: 500 },
        steps: 100_000,
        mode: 'eager',
        seed: 5,
      })
      expect(s.status).toBe('converged')
      const post = m.posterior?.(s.state, s.config)
      expect(post?.draws?.['z']?.shape).toEqual([1500, 3])
      const table = agree([fromPosterior('hybrid', post as never, { sites: ['locs'] })], {
        oracle,
        quantities: Object.keys(oracle),
        quantilePoints: 300,
      })
      expect(
        agreementWithin(table, { meanError, meanZ: 8, sdRatio: [0.7, 1.4], ks: 0.15 }).map(
          (b) =>
            `${b.row.quantity}: ${b.failed.join(',')} (mean error ${b.row.meanError.toFixed(2)}, sd ratio ${b.row.sdRatio.toFixed(2)})`,
        ),
      ).toEqual([])
    }
  })

  it('draws a discrete site that a continuous block sums out, from its conditional, when a draw is kept', () => {
    const model = DISCRETE['mixture'] as Model
    const m = hybridMethod(model, { blocks: [{ sites: ['locs'], kernel: blockKernels.nuts(), marginalise: ['z'] }] })
    expect(m.check?.({ latent: ['locs', 'z'], continuous: ['locs'] } as never)).toMatchObject({ applies: true })
    const s = session(m, { config: { numChains: 2, numWarmup: 50, numDraws: 40 }, steps: 1000, mode: 'eager' })
    const z = m.posterior?.(s.state, s.config).draws?.['z']
    expect(z?.shape).toEqual([80, 3])
    expect(Array.from(z?.data ?? []).every((v) => v === 0 || v === 1)).toBe(true)
  })
})

describe('the plan and the method convention', () => {
  const small = { numChains: 2, numWarmup: 25, numDraws: 15 }
  const blocks: HybridBlock[] = [
    { sites: ['a'], kernel: blockKernels.nuts() },
    { sites: ['b'], kernel: blockKernels.gibbs() },
  ]

  it('says what is wrong with a plan: an unassigned site, an unknown one, a discrete site on a gradient kernel', () => {
    const a = { latent: ['a', 'b'], continuous: ['a', 'b'] } as never
    expect(hybridMethod(chain, { blocks: [blocks[0] as HybridBlock] }).check?.(a)).toMatchObject({
      applies: false,
      missing: ['a block for every latent site'],
    })
    expect(
      hybridMethod(chain, { blocks: [{ sites: ['a', 'c'], kernel: blockKernels.nuts() }] }).check?.(a),
    ).toMatchObject({ applies: false })
    expect(
      hybridMethod(chain, {
        blocks: [
          { sites: ['a'], kernel: blockKernels.nuts() },
          { sites: ['a'], kernel: blockKernels.gibbs() },
        ],
      }).check?.(a).applies,
    ).toBe(false)
    const disc = { latent: ['z'], continuous: [] } as never
    expect(hybridMethod(chain, { blocks: [{ sites: ['z'], kernel: blockKernels.hmc() }] }).check?.(disc)).toMatchObject(
      { missing: ['continuous sites'] },
    )
    expect(hybridMethod(chain, { blocks: [] }).check?.(a).applies).toBe(false)
    expect(() => session(hybridMethod(chain, { blocks: [blocks[0] as HybridBlock] }), { steps: 3 })).toThrow(
      /no block moves b/,
    )
  })

  it('puts the block statistics on the tape, cancels within one sweep, resumes exactly and snapshots', () => {
    const m = hybridMethod(chain, { blocks })
    const s = session(m, { config: small, steps: 500, mode: 'eager', logs: 'all' })
    const paths = new Set(s.run().events.flatMap((e) => ('name' in e ? [e.name] : [])))
    for (const p of [
      'hybrid/accept',
      'hybrid/stepSize',
      'hybrid/logJoint',
      'hybrid/kept',
      'hybrid/stage',
      'hybrid/order',
    ])
      expect(paths.has(p), p).toBe(true)
    expect(s.t).toBe(1 + small.numWarmup + small.numDraws)
    expect(s.probe(s.t, 'hybrid/snapshot')).toMatchObject({ engine: 'hybrid', status: 'completed' })
    expectCancelAndContinue(m, { config: small, steps: 500, cancelAfterSlices: 7 })
    expectResumeEqual(m, { config: small, steps: 500, at: 12 })
    expectResumeEqual(m, { config: small, steps: 500, at: 1 + small.numWarmup + 5 })
    expectSnapshotCadence(m, {
      config: { ...small, numDraws: 300 },
      steps: 5000,
      budgetMs: 2500,
      everyMs: 1000,
      tick: 100,
    })
  })

  it('loads a state saved with the kept draws as a typed array, before they were tensors', () => {
    const m = hybridMethod(chain, { blocks })
    const s = session(m, { config: small, steps: 500, mode: 'eager' })
    const algorithm = m.make(s.config, key(0))
    const saved = JSON.parse(JSON.stringify(algorithm.serialise?.(s.state))) as { chains: Record<string, Json>[] }
    const older = {
      ...saved,
      chains: saved.chains.map((ch) => {
        const flat = Float64Array.from(toFlat(stateFromJson(ch['kept'] as Json) as Tensor))
        return { ...ch, kept: stateToJson(flat) }
      }),
    }
    const back = algorithm.deserialise?.(older as never)
    expect(back?.chains.map((ch) => ch.kept.map((r) => toFlat(r)))).toEqual(
      s.state.chains.map((ch) => ch.kept.map((r) => toFlat(r))),
    )
  })

  it('visits the blocks in a random order when asked', () => {
    const m = hybridMethod(chain, { blocks })
    const s = session(m, { config: { ...small, order: 'random' }, steps: 500, mode: 'eager', logs: 'all' })
    const orders = s.run().events.filter((e) => 'name' in e && e.name === 'hybrid/order')
    expect(orders.length).toBeGreaterThan(5)
  })
})
