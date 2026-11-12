import { scan } from 'omai-engine/nn'
import { epSpecMethod, runAnytimeAsync } from 'omai-engine/probprog/inference'
import { defineModel } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { key, normal, split, uniform } from 'omai-engine/random'
import { add, concat, less, mul, reshape, sliceAxis, take, toFlat, where } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
import type { EpHooks } from '../../../../shared/ep/sweeps.ts'

export const PATH = { phi: 0.97, sigma: 0.3, start: 1.5 } // x_t = phi x_(t-1) + N(0, sigma^2)
export const WIDTH = 8 // an outlier's noise is 8 times as wide as an ordinary one

// region simulate
type Noise = { sd: number; outliers: number }
/**
 * The truth: an AR(1) path, seen with noise of sd `sd`.
 * A fraction `outliers` of the readings is wild (8 times as noisy).
 */
export function simulate(seed: number, noise: Noise, steps = 100) {
  type Key = ReturnType<typeof key>
  const [kx, ku, ke] = split(key(seed), 3) as [Key, Key, Key]
  const w = normal(kx, [steps + 1])
  const start = mul(take(w, 0), PATH.start)
  // Each state follows the one before it.
  const path = scan(
    (prev, noise) => {
      const next = add(mul(prev, PATH.phi), mul(noise, PATH.sigma))
      return [next, next]
    },
    start,
    sliceAxis(w, 0, 1, steps),
  )
  const x = concat([reshape(start, [1]), path.ys])
  const wild = less(uniform(ku, [steps]), noise.outliers)
  const e = normal(ke, [steps])
  const y = add(x, mul(mul(e, noise.sd), where(wild, WIDTH, 1)))
  return { x: toFlat(x), y: toFlat(y), wild: toFlat(wild), noise }
}
// endregion

// region model
/**
 * x_t = phi x_(t-1) + noise, a scan; each reading is the state plus noise.
 * A fraction `outliers` of the readings is wild (or the noise is plain Gaussian).
 */
export function model({ sd, outliers }: Noise, robust = true) {
  return defineModel('tracker', (b) => {
    const y = b.data('y', { dims: ['time'] })
    b.scan('x', {
      over: 'time',
      init: d.normal(0, PATH.start),
      step: ({ prev }) => d.normal(prev.mul(PATH.phi), PATH.sigma),
      emit: ({ state, observe }) => {
        // A fraction of the readings is wild, or the noise is plain Gaussian.
        const reading = robust ? d.normalOutliers(state, sd, outliers, WIDTH) : d.normal(state, sd)
        return observe('y', reading, y)
      },
    })
  })
}
// endregion

// region infer
/** Expectation propagation on the chain: the state at each step, with 90% intervals. */
export async function infer(y: readonly number[], noise: Noise, robust = true, hooks: EpHooks = {}) {
  const bound = model(noise, robust).bind({ y })
  const ep = epSpecMethod(bound)
  const config = { tol: 1e-9 }
  const cap = 300
  const run = session(ep, { config, seed: 1, steps: cap, mode: 'live', logs: 'all' })
  // One sweep at a time: the page hears of each (the bar), and may stop the run.
  await runAnytimeAsync(run, {
    snapshot: { everySteps: 1 },
    onSnapshot: ({ t, state }) => hooks.onSweep?.({ sweep: t, cap, change: state.change, tol: config.tol }),
    shouldStop: () => hooks.stopped?.() === true,
    yieldToPage: hooks.yieldToPage ?? (() => Promise.resolve()),
  })
  const rows =
    ep
      .posterior?.(run.state, run.config)
      ?.summary({ prob: 0.9 })
      .filter((r) => r.site.startsWith('x[')) ?? []
  return { bound, run, rows }
}
// endregion
