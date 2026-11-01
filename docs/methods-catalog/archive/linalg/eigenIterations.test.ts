import { describe, expect, it } from 'vitest'
import { metricSeries, session } from '../../lib/trace/index.ts'
import { eigenPower, eigenQr } from './eigenIterations.ts'

describe('eigen iterations', () => {
  it('the power iteration converges to the leading eigenvalue', () => {
    const s = session(eigenPower, { mode: 'eager', steps: 80, config: { n: 4, spread: 4 } })
    const error = metricSeries(s.run(), 'iteration/error').values
    expect(error.at(-1)).toBeLessThan(1e-6)
    const action = s.probe(40, 'matrix/action') as { readonly series: { name: string }[] }
    expect(action.series.map((x) => x.name)).toEqual(['unit circle', 'its image under A', 'v', 'A v'])
  })

  it('the QR iteration diagonalises the matrix', () => {
    const s = session(eigenQr, { mode: 'eager', steps: 80, config: { n: 4, spread: 4 } })
    const off = metricSeries(s.run(), 'iteration/offDiagonal').values
    expect(off.at(-1)).toBeLessThan(1e-6)
    expect(metricSeries(s.run(), 'iteration/error').values.at(-1)).toBeLessThan(1e-6)
  })
})
