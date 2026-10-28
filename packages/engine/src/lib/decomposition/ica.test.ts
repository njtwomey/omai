import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { fastIca, fastIcaAlgorithm, fastIcaDeflationAlgorithm, symmetricDecorrelation } from './ica.ts'

const t = Array.from({ length: 200 }, (_, i) => i / 12)
const sources = t.map((v) => [Math.sin(2 * v), Math.sign(Math.sin(3.1 * v)), (v % 1.3) - 0.65])
const mix = [
  [1, 0.5, 0.2],
  [0.4, 1, 0.3],
  [0.2, 0.6, 1],
]
const x = sources.map((s) => mix.map((row) => row.reduce((a, m, j) => a + m * (s[j] ?? 0), 0)))
const corr = (a: number[], b: number[]): number => {
  const ma = a.reduce((s, v) => s + v, 0) / a.length
  const mb = b.reduce((s, v) => s + v, 0) / b.length
  let ab = 0
  let aa = 0
  let bb = 0
  for (const [i, v] of a.entries()) {
    ab += (v - ma) * ((b[i] ?? 0) - mb)
    aa += (v - ma) ** 2
    bb += ((b[i] ?? 0) - mb) ** 2
  }
  return ab / Math.sqrt(aa * bb)
}

describe('FastICA', () => {
  for (const algorithm of ['parallel', 'deflation'] as const)
    it(`${algorithm}: recovers the sources up to order and sign, and inverts the transform`, () => {
      const f = fastIca.with({ nComponents: 3, algorithm }).fit({ x }, { key: key(1), logs: 'all' })
      const s = toFlat(f.transform(x))
      const cols = [0, 1, 2].map((c) => s.filter((_, i) => i % 3 === c))
      for (let j = 0; j < 3; j++) {
        const truth = sources.map((r) => r[j] ?? 0)
        expect(Math.max(...cols.map((c) => Math.abs(corr(c, truth))))).toBeGreaterThan(0.95)
      }
      const back = toFlat(f.inverseTransform(f.transform(x)))
      const flat = x.flat()
      for (const [i, v] of back.entries()) expect(v).toBeCloseTo(flat[i] ?? 0, 6)
      if (algorithm === 'parallel') expect(metricSeries(f.run, 'solver/change').values.length).toBeGreaterThan(1)
    })

  it('a session shows the deflation finding one component per step', () => {
    const s = fastIca.with({ nComponents: 3, algorithm: 'deflation' }).session({ x }, { key: key(1), logs: 'all' })
    s.advance({ steps: 1 })
    expect(s.fittedAt(1).params.components.length).toBe(1)
    expect((s.probe(1, 'model/embedding') as number[][])[0]?.length).toBe(1)
  })

  it('states round-trip; decorrelation makes the rows orthonormal', () => {
    const z = [
      [1, -1, 0.5, -0.5],
      [0.5, 0.5, -1, 1],
    ]
    const p = runAlgorithm(
      fastIcaAlgorithm(z, [
        [1, 0.2],
        [0.1, 1],
      ]),
      { maxSteps: 3, key: key(0) },
    ).state
    expect(stateFromJson(stateToJson(p))).toEqual(p)
    const d = runAlgorithm(
      fastIcaDeflationAlgorithm(z, [
        [1, 0.2],
        [0.1, 1],
      ]),
      { maxSteps: 2, key: key(0) },
    ).state
    expect(stateFromJson(stateToJson(d))).toEqual(d)
    const w = symmetricDecorrelation([
      [2, 1],
      [0, 3],
    ])
    expect(w[0]?.reduce((a, v) => a + v * v, 0)).toBeCloseTo(1, 12)
    expect(w[0]?.reduce((a, v, j) => a + v * (w[1]?.[j] ?? 0), 0)).toBeCloseTo(0, 12)
  })
})
