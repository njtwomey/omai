import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { inductionHeads, selectiveCopying } from '../datasets/index.ts'
import { sequenceMixers } from '../nn/blocks/index.ts'
import { key } from '../random/index.ts'
import { metricSeries } from '../trace/index.ts'
import type { LearnData } from './estimator.ts'
import { sequenceModelTrainer, tokenAccuracy } from './sequence-model.ts'

const copying = (n: number, seed: number): LearnData => {
  const { data } = generate(selectiveCopying, { knobs: { n, length: 10, tokens: 3, symbols: 5 }, key: key(seed) })
  return { x: data.inputs, y: data.targets }
}

describe('token accuracy', () => {
  it('counts the positions that carry a target only', () => {
    expect(
      tokenAccuracy(
        [
          [
            [0, 1],
            [1, 0],
            [0, 1],
          ],
        ],
        [[0, 1, 1]],
      ),
    ).toBe(0.5)
  })
})

describe('the sequence-model trainer', () => {
  it('a Mamba layer learns selective copying that a fixed-kernel state-space layer does not (counted accuracy)', () => {
    // Three content symbols among ten positions, five symbols: chance is 1/5. Same width, depth, optimiser, steps
    // and seeds; the time-invariant layer (S4D) cannot choose what to keep by what the input says.
    const train = copying(512, 1)
    const fresh = copying(384, 9)
    const accuracy = (mixer: { kind: string; stateSize: number }, steps: number): number => {
      const fit = sequenceModelTrainer({ mixer, dim: 12, layers: 1, steps, batchSize: 16, learningRate: 0.02 }).fit(
        train,
        { key: key(1), explain: false },
      )
      return tokenAccuracy(fit.predict?.(fresh.x) ?? [], fresh.y ?? [])
    }
    const mamba = accuracy({ kind: 'mamba', stateSize: 6 }, 200)
    const s4d = accuracy({ kind: 's4d', stateSize: 6 }, 200)
    expect(mamba).toBeGreaterThan(0.85)
    expect(s4d).toBeLessThan(0.75)
    expect(mamba - s4d).toBeGreaterThan(0.15)
  }, 120_000)

  it('logs the loss, the validation accuracy, and probes the spectrum, the learned steps and the state norms', () => {
    const data = copying(160, 2)
    const est = sequenceModelTrainer({
      mixer: { kind: 'mamba', stateSize: 4 },
      dim: 8,
      steps: 12,
      batchSize: 16,
      validationFraction: 0.2,
    })
    const fit = est.fit(data, { key: key(3) })
    expect(metricSeries(fit.run, 'train/loss').values).toHaveLength(12)
    expect(metricSeries(fit.run, 'eval/accuracy').values.length).toBeGreaterThan(0)
    const s = est.session(data, { key: key(3), mode: 'live' })
    s.advance({ steps: 4 })
    expect(Object.keys(s.contract)).toEqual(
      expect.arrayContaining(['mixer/spectrum', 'mixer/step', 'mixer/state-norm']),
    )
    const spectrum = s.probe(4, 'mixer/spectrum') as Record<string, { domain: string; re: number[] }>
    expect(spectrum['layers.0']?.domain).toBe('continuous')
    expect(spectrum['layers.0']?.re.every((v) => v < 0)).toBe(true)
    const steps = s.probe(4, 'mixer/step') as Record<string, number>
    expect(steps['layers.0.mixer.dt']).toBeGreaterThan(0)
    const norms = s.probe(4, 'mixer/state-norm') as Record<string, number>
    expect(Number.isFinite(norms['layers.0'])).toBe(true)
  })

  it('takes the mixer and the optimiser from their tables: every mixer trains a step', () => {
    const { data } = generate(inductionHeads, { knobs: { n: 32, length: 8, symbols: 4 }, key: key(4) })
    const rows: LearnData = { x: data.inputs, y: data.targets }
    for (const kind of sequenceMixers.names) {
      const est = sequenceModelTrainer({ dim: 8, steps: 2, batchSize: 16, optimiser: { kind: 'sgd' } }).with({
        mixer: { kind },
      })
      expect(est.hyper.mixer).toMatchObject({ kind })
      const loss = metricSeries(est.fit(rows, { key: key(5) }).run, 'train/loss').values
      expect(loss).toHaveLength(2)
      expect(loss.every(Number.isFinite)).toBe(true)
    }
  }, 120_000)
})
