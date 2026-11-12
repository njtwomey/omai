import { scan } from 'omai-engine/nn'
import { epSpecMethod, runAnytimeAsync } from 'omai-engine/probprog/inference'
import { defineModel, ops } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { key, normal, split } from 'omai-engine/random'
import { add, concat, expandDims, mul, sliceAxis, stack, take, tensor, toFlat, type Value } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
import type { EpHooks } from '../../../../shared/ep/sweeps.ts'

/** The time between readings, the start of the state and what the page changes about the data. */
export const DT = 0.5
export const START: { readonly mean: readonly [number, number]; readonly sd: readonly [number, number] } = {
  mean: [0, 1],
  sd: [2, 1],
} // the position and the velocity at the first step
export interface Setup {
  readonly seed: number
  readonly steps: number
  /** The sd of each position reading. */
  readonly readingSd: number
  /** How strongly the velocity is pushed around (the intensity of the white acceleration). */
  readonly push: number
}
export const defaultSetup: Setup = { seed: 3, steps: 60, readingSd: 0.6, push: 0.4 }

/** The covariance of the push in one step: a white acceleration of intensity `push`, which moves the position and the velocity together. */
export const stepCovariance = (push: number): number[][] => [
  [(push * DT ** 3) / 3, (push * DT ** 2) / 2],
  [(push * DT ** 2) / 2, push * DT],
]

// region simulate
/** The truth: a position and a velocity pushed around, the position read with noise. */
export function simulate({ seed, steps, readingSd, push }: Setup) {
  type Key = ReturnType<typeof key>
  const [kx, kw, ke] = split(key(seed), 3) as [Key, Key, Key]
  const start = normal(kx, [2])
  const w = normal(kw, [steps, 2])
  // The push is correlated between the position and the velocity: its Cholesky factor mixes two standard normals.
  const q = stepCovariance(push)
  const l00 = Math.sqrt(q[0]?.[0] as number)
  const l10 = (q[1]?.[0] as number) / l00
  const l11 = Math.sqrt((q[1]?.[1] as number) - l10 ** 2)
  const first = add(tensor(START.mean), mul(tensor(START.sd), start))
  // The position moves by the velocity and the push; the velocity by the push.
  const path = scan(
    ([p, v], shock) => {
      const a = take(shock, 0)
      const b = take(shock, 1)
      const position = add(add(p, mul(v, DT)), mul(a, l00))
      const velocity = add(add(v, mul(a, l10)), mul(b, l11))
      return [[position, velocity] as [Value, Value], stack([position, velocity])]
    },
    [take(first, 0), take(first, 1)] as [Value, Value],
    sliceAxis(w, 0, 1, steps),
  )
  const states = concat([expandDims(first, 0), path.ys], 0)
  const position = take(states, 0, 1)
  const y = add(position, mul(normal(ke, [steps]), readingSd))
  return { position: toFlat(position), velocity: toFlat(take(states, 1, 1)), y: toFlat(y) }
}
// endregion

// region model
/** A position and a velocity, a scan over time with a vector state; each reading is the position plus noise. */
export function model({ readingSd, push }: Pick<Setup, 'readingSd' | 'push'>) {
  return defineModel('tracker-2d', (m) => {
    const y = m.data('y', { dims: ['time'] })
    // the position moves by the velocity in each step, the velocity stays
    const step = m.const([
      [1, DT],
      [0, 1],
    ])
    // a reading sees the position only
    const position = m.const([1, 0])
    const start = d.multivariateNormal(START.mean, {
      covariance: [
        [START.sd[0] ** 2, 0],
        [0, START.sd[1] ** 2],
      ],
    })
    m.scan('x', {
      over: 'time',
      init: start,
      step: ({ prev }) => d.multivariateNormal(ops.matvec(step, prev), { covariance: stepCovariance(push) }),
      emit: ({ state, observe }) => observe('y', d.normal(ops.dot(position, state), readingSd), y),
    })
  })
}
// endregion

// region infer
/** Expectation propagation on the chain: the position and the velocity at each step, with 90% intervals. */
export async function infer(y: readonly number[], setup: Setup, hooks: EpHooks = {}) {
  const bound = model(setup).bind({ y })
  const ep = epSpecMethod(bound)
  const config = { tol: 1e-10 }
  const cap = 300
  const run = session(ep, { config, seed: 1, steps: cap, mode: 'live', logs: 'all' })
  // One sweep at a time: the page hears of each (the bar), and may stop the run.
  await runAnytimeAsync(run, {
    snapshot: { everySteps: 1 },
    onSnapshot: ({ t, state }) => hooks.onSweep?.({ sweep: t, cap, change: state.change, tol: config.tol }),
    shouldStop: () => hooks.stopped?.() === true,
    yieldToPage: hooks.yieldToPage ?? (() => Promise.resolve()),
  })
  const rows = ep.posterior?.(run.state, run.config)?.summary({ prob: 0.9 }) ?? []
  // The coordinates of each step are rows named x[t][0] (the position) and x[t][1] (the velocity).
  const of = (k: 0 | 1) => rows.filter((r) => r.name.endsWith(`][${k}]`))
  return { bound, run, position: of(0), velocity: of(1) }
}
// endregion
