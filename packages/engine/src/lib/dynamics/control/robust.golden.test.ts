import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../../testing/systems.ts'
import { evalfr } from '../../systems/index.ts'
import { fromStorage, item, type Tensor, toFlat } from '../../tensor/index.ts'
import { hinfSynthesis } from './robust.ts'

const FNS: Readonly<Record<string, GoldenFn>> = {
  'hinf-controller': ([a, b1, b2, c1, c2, d12, d21, w], p) => {
    const k = hinfSynthesis({ a, b1, b2, c1, c2, d12, d21 }, p['gamma'] as number).controller
    if (k === undefined) throw new Error('the level should be achievable')
    const freqs = toFlat(w as Tensor)
    const re: number[] = []
    const im: number[] = []
    for (const f of freqs) {
      const z = evalfr(k, { re: 0, im: f })
      re.push(item(z.re))
      im.push(item(z.im))
    }
    return fromStorage(Float64Array.from([...re, ...im]), [2, freqs.length], 'float64')
  },
}

describe('dynamics/control/robust against SLICOT (slycot)', () => {
  runFixture('dynamics/control/robust', FNS, it)
})
