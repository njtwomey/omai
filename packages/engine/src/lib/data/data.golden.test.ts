import { expect, describe as group, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, tensor } from '../tensor/index.ts'
import { balancedWeights, describe, padSequences, table, tableToXy, xy } from './index.ts'

group('data against numpy, scikit-learn and torch', () => {
  for (const c of loadFixture('data/tensors').cases) {
    it(c.name, () => {
      if (c.fn === 'describe') {
        const x = decode(c.inputs[0]) as Tensor
        const s = describe(xy(x, new Array<number>(x.shape[0] ?? 0).fill(0)))
        const rows = s.features.map((f) => [f.count, f.missing, f.mean, f.std, f.min, f.q25, f.median, f.q75, f.max])
        expectMatches(tensor(rows), c.output, c.rtol, c.atol, c.name)
        return
      }
      const colours = String(c.params['colours']).split(',')
      const d = tableToXy(table({ colour: colours, y: colours.map((_, i) => i) }, { target: 'y' }), {
        keepFirst: c.params['drop'] !== 1,
      })
      expectMatches(d.x, c.output, c.rtol, c.atol, c.name)
    })
  }
  for (const c of loadFixture('data/batching').cases) {
    it(c.name, () => {
      if (c.fn === 'balancedWeights') {
        expectMatches(balancedWeights(decode(c.inputs[0]) as Tensor), c.output, 1e-12, 1e-12, c.name)
        return
      }
      const seqs =
        c.inputs.length === 1 && Array.isArray(c.inputs[0])
          ? (c.inputs[0] as number[][])
          : c.inputs.map((s) => decode(s) as Tensor)
      const p = padSequences(seqs, {
        padValue: c.params['padValue'] as number,
        padding: (c.params['padding'] as 'pre' | undefined) ?? 'post',
        dtype: 'float64',
      })
      expectMatches(p.values, c.output, c.rtol, c.atol, c.name)
    })
  }
  it('every case ran', () => {
    expect(loadFixture('data/tensors').cases.length).toBeGreaterThan(0)
  })
})
