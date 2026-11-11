import { epSpecMethod } from 'omai-engine/probprog/inference'
import { defineModel } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { key, normal, split, uniform } from 'omai-engine/random'
import { toFlat } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'

export const PATH = { phi: 0.97, sigma: 0.3, start: 1.5 } // x_t = phi x_(t-1) + N(0, sigma^2)
export const WIDTH = 8 // an outlier's noise is 8 times as wide as an ordinary one

// region simulate
type Noise = { sd: number; outliers: number }
/** The truth: an AR(1) path, seen with noise of sd `sd`, and a fraction `outliers` of the readings wild (8 times as noisy). */
export function simulate(seed: number, noise: Noise, steps = 100) {
  const [kx, ku, ke] = split(key(seed), 3) as [ReturnType<typeof key>, ReturnType<typeof key>, ReturnType<typeof key>]
  const w = toFlat(normal(kx, [steps + 1]))
  const x = [PATH.start * (w[0] ?? 0)]
  for (let t = 1; t < steps; t++) x.push(PATH.phi * (x[t - 1] as number) + PATH.sigma * (w[t] ?? 0))
  const wild = toFlat(uniform(ku, [steps])).map((u) => u < noise.outliers)
  const e = toFlat(normal(ke, [steps]))
  const y = x.map((v, t) => v + (e[t] ?? 0) * noise.sd * (wild[t] ? WIDTH : 1))
  return { x, y, wild, noise }
}
// endregion

// region model
/** x_t = phi x_(t-1) + noise, a scan; each reading is the state plus noise, with a fraction `outliers` of them wild (or plain Gaussian). */
export function model({ sd, outliers }: Noise, robust = true) {
  return defineModel('tracker', (b) => {
    const y = b.data('y', { dims: ['time'] })
    b.scan('x', {
      over: 'time',
      init: d.normal(0, PATH.start),
      step: ({ prev }) => d.normal(prev.mul(PATH.phi), PATH.sigma),
      emit: ({ state, observe }) =>
        observe('y', robust ? d.normalOutliers(state, sd, outliers, WIDTH) : d.normal(state, sd), y),
    })
  })
}
// endregion

// region infer
/** Expectation propagation on the chain: the posterior of the state at each step, with 90% intervals. */
export function infer(y: readonly number[], noise: Noise, robust = true) {
  const bound = model(noise, robust).bind({ y })
  const ep = epSpecMethod(bound)
  const run = session(ep, { config: { tol: 1e-9 }, seed: 1, steps: 300, mode: 'eager', logs: 'all' })
  const rows =
    ep
      .posterior?.(run.state, run.config)
      ?.summary({ prob: 0.9 })
      .filter((r) => r.site.startsWith('x[')) ?? []
  return { bound, run, rows }
}
// endregion
