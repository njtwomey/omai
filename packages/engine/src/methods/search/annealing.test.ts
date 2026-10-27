import { describe, expect, it } from 'vitest'
import { metricSeries, session, tensorSeries } from '../../lib/trace/index.ts'
import { annealing } from './annealing.ts'

describe('annealing', () => {
  it('walks over Himmelblau, cooling, with the objective as a probe that survives seeking', () => {
    const s = session(annealing, { mode: 'eager', steps: 400, config: { objective: 'himmelblau', cooling: 0.99 } })
    const run = s.run()
    // A position (two numbers) per step, kept in full.
    const walk = tensorSeries(run, 'position')
    expect(walk.length).toBe(400)
    expect(walk[0]?.values?.length).toBe(2)
    const best = metricSeries(run, 'best').values
    expect(best.at(-1) ?? Number.NaN).toBeLessThan(best[0] ?? Number.NaN)
    expect(best.every((v, i) => i === 0 || v <= (best[i - 1] ?? Number.NaN))).toBe(true)
    const temperature = metricSeries(run, 'temperature').values
    expect(temperature[0]).toBeGreaterThan(temperature.at(-1) ?? Number.NaN)
    // The surface is the objective on a 60 by 60 grid, wherever the walk is.
    const surface = s.probe(120, 'objective/surface') as { kind: string; values: number[][]; x: number[] }
    expect(surface.kind).toBe('decisionSurface')
    expect(surface.values.length).toBe(60)
    expect(surface.values[0]?.length).toBe(60)
    expect(surface.x[0]).toBe(-5)
    // Seeking rebuilds the state from a checkpoint: the walk is back where the tape says it was.
    const there = s.stateAt(200).current as { x: number; y: number }
    const at = walk.find((e) => e.step === 199)?.values
    expect([there.x, there.y]).toEqual(at)
  })

  it('climbs to a local minimum and stops', () => {
    const s = session(annealing, {
      mode: 'eager',
      steps: 400,
      config: { algorithm: 'hill-climbing', objective: 'himmelblau' },
    })
    expect(s.status).toBe('stopped')
    expect(s.t).toBeLessThan(400)
    const energy = metricSeries(s.run(), 'energy').values
    expect(energy.at(-1) ?? Number.NaN).toBeLessThan(energy[0] ?? Number.NaN)
  })
})
