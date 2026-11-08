import { generate } from 'omai-engine/data'
import { piecewiseLinear } from 'omai-engine/datasets'
import { describe, expect, it } from 'vitest'
import { key } from '../../lib/random/index.ts'
import { toArray, toFlat } from '../../lib/tensor/index.ts'
import { moe } from './moe.ts'

describe('moe', () => {
  const { data } = generate(piecewiseLinear, { knobs: { n: 80, knots: 1 }, key: key(1) })

  it('fits two experts, predicts, and runs as a session with its probes', () => {
    const fitted = moe.fit(data, { key: key(0) })
    expect(fitted.params.experts.length).toBe(2)
    expect(toFlat(fitted.predict(data.x)).length).toBe(80)
    for (const row of toArray(fitted.predictGate(data.x)) as number[][])
      expect(row.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 9)

    const s = moe.session(data, { key: key(0), logs: 'defaults' })
    s.advance({ steps: 6 })
    expect(s.t).toBeGreaterThan(0)
    const assignments = s.probe(s.t, 'model/assignments') as number[]
    expect(assignments.length).toBe(80)
    const experts = s.probe(s.t, 'model/experts') as { lines: number[][] }
    expect(experts.lines.length).toBe(2)
    expect(experts.lines[0]?.length).toBe(80)
    const gating = s.probe(s.t, 'model/gating') as { kind: string; series: { y: number[] }[] }
    expect(gating.kind).toBe('curves')
    expect(gating.series.length).toBe(2)
    expect((s.probe(s.t, 'model/predictions') as number[]).length).toBe(80)
  })
})
