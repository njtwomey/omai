import { epSpecMethod } from 'omai-engine/probprog/inference'
import { defineModel, ops } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { foldIn, key, poisson, split, uniform } from 'omai-engine/random'
import { toFlat } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'

export const RATES = [3, 9, 22] // the truth: the rate of each regime
export const PRIOR = { shape: [2, 2, 2], rate: [0.5, 1 / 6, 1 / 15] } // the model's: means 4, 12 and 30, each broad (a low, a middle and a high regime)

// region simulate
/** A sticky chain over the regimes (stay with probability `stay`, else move to either other), and Poisson counts at each regime's rate. */
export function simulate(seed: number, stay = 0.95, steps = 80) {
  const [kz, ky] = split(key(seed), 2) as [ReturnType<typeof key>, ReturnType<typeof key>]
  const u = toFlat(uniform(kz, [steps]))
  const z: number[] = [Math.floor(3 * (u[0] ?? 0))]
  for (let t = 1; t < steps; t++) {
    const prev = z[t - 1] as number
    const move = ((u[t] ?? 0) - stay) / (1 - stay) // in [0, 1) when it moves
    z.push(move < 0 ? prev : (prev + 1 + Math.floor(2 * move)) % 3)
  }
  const y = z.map((s, t) => toFlat(poisson(foldIn(ky, t), RATES[s] as number, [1]))[0] ?? 0)
  return { z, y, stay }
}
// endregion

// region model
/** z: a sticky Markov chain over three regimes; each regime has an unknown rate (gamma prior), the count at t is Poisson(rate[z_t]). */
export function model(stay: number) {
  return defineModel('poisson-changepoint', (b) => {
    const y = b.data('y', { dims: ['time'], support: 'count' })
    const regime = b.range('regime', { size: 3 })
    const move = (1 - stay) / 2
    const A = b.const(
      [
        [stay, move, move],
        [move, stay, move],
        [move, move, stay],
      ],
      ['from'],
    )
    const rate = b.latent('rate', d.gamma(b.const(PRIOR.shape, ['regime']), b.const(PRIOR.rate, ['regime'])), {
      dims: [regime],
    })
    b.scan('z', {
      over: 'time',
      init: d.categorical({ probs: [1 / 3, 1 / 3, 1 / 3] }),
      step: ({ prev }) => d.categorical({ probs: ops.at(A, prev) }),
      emit: ({ state, observe }) => observe('y', d.poisson(ops.at(rate, state)), y),
    })
  })
}
// endregion

// region infer
/** Expectation propagation on the factor graph: the regime probabilities per step and the rates. */
export function infer(y: readonly number[], stay: number) {
  const bound = model(stay).bind({ y })
  const ep = epSpecMethod(bound)
  const run = session(ep, { config: { tol: 1e-9 }, seed: 1, steps: 300, mode: 'eager', logs: 'all' })
  const post = ep.posterior?.(run.state, run.config)
  const regimes = Array.from({ length: y.length }, (_, t) =>
    [0, 1, 2].map((k) => toFlat(post?.marginal(`z[${t}]`).prob(k) as never)[0] ?? 0),
  )
  const rates = post?.summary({ prob: 0.9 }).filter((r) => r.site === 'rate') ?? []
  return { bound, run, regimes, rates }
}
// endregion
