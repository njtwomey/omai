import { describe, expect, it } from 'vitest'
import { adam } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { toArray } from '../tensor/index.ts'
import * as D from './index.ts'

const schedule = D.linearSchedule()
const modes = [
  [-1.5, -1.5],
  [1.5, -1.5],
  [-1.5, 1.5],
  [1.5, 1.5],
]
const truth = D.gaussianMixtureTruth({ means: modes, stds: 0.25 })

/** Train an epsilon network on the mixture and return it with its final parameters and the loss history. */
function train(
  steps: number,
  seed: number,
): { network: D.ScoreNetwork; params: ReturnType<typeof networkParams>; losses: number[] } {
  const network = D.scoreNetwork({ dim: 2, hidden: [64, 64], key: key(seed) })
  const trainer = D.diffusionTrainer({
    schedule,
    data: truth.sample(key(100), 1500).x,
    network,
    optimizer: adam({ learningRate: 3e-3 }),
    batchSize: 128,
    steps,
  })
  const losses: number[] = []
  let state = trainer.init(key(seed))
  for (let i = 0; i < steps; i++) {
    state = trainer.step(state, {
      t: i,
      key: key(i),
      run: undefined as never,
      log: (path: string, value: unknown) => {
        if (path === 'diffusion/loss') losses.push(value as number)
      },
      wants: () => true,
      emit: () => undefined,
      taps: undefined as never,
    })
  }
  return { network, params: networkParams(state), losses }
}
const networkParams = (state: { params: unknown }): D.ScoreNetwork['module']['params'] =>
  state.params as D.ScoreNetwork['module']['params']

describe('training a diffusion network on a toy mixture', () => {
  it('learns it: the loss falls, and samples land on the modes of the data', () => {
    const { network, params, losses } = train(700, 1)
    const head = losses.slice(0, 50).reduce((a, v) => a + v, 0) / 50
    const tail = losses.slice(-50).reduce((a, v) => a + v, 0) / 50
    expect(tail).toBeLessThan(head * 0.7)
    const quality = (model: D.DiffusionModel): { near: number; counts: number[] } => {
      const { x } = D.sampleDiffusion({ schedule, model, shape: [500, 2], steps: 40, method: 'ddim' }, key(5))
      const counts = [0, 0, 0, 0]
      let near = 0
      const rows = toArray(x) as number[][]
      for (const r of rows) {
        const d = modes.map((c) => Math.hypot((r[0] ?? 0) - (c[0] ?? 0), (r[1] ?? 0) - (c[1] ?? 0)))
        const k = d.indexOf(Math.min(...d))
        counts[k] = (counts[k] ?? 0) + 1
        if ((d[k] ?? 9) < 1) near += 1
      }
      return { near: near / rows.length, counts: counts.map((c) => c / rows.length) }
    }
    const trained = quality(D.networkModel(network, params))
    // The untrained network is far from the data; the exact score is the ceiling.
    const untrained = quality(D.networkModel(network, network.module.params))
    const exact = quality(truth.model(schedule, 'epsilon'))
    expect(untrained.near).toBeLessThan(0.3)
    expect(exact.near).toBeGreaterThan(0.95)
    expect(trained.near).toBeGreaterThan(0.7)
    for (const c of trained.counts) expect(c).toBeGreaterThan(0.1)
  }, 30_000)
})
