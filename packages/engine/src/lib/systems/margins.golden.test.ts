import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../testing/systems.ts'
import { type Tensor, tensor } from '../tensor/index.ts'
import * as S from './index.ts'

const loop = (n: Tensor | undefined, d: Tensor | undefined, dt: unknown): S.TransferFunction =>
  S.tf(n as Tensor, d as Tensor, { dt: dt as number })

const FNS: Readonly<Record<string, GoldenFn>> = {
  stabilityMargins: ([n, d], p) => {
    const m = S.stabilityMargins(loop(n, d, p['dt']))
    return tensor([m.gm, m.pm, m.sm, m.wpc, m.wgc, m.wms])
  },
  allMargins: ([n, d], p) => tensor([...S.allMargins(loop(n, d, p['dt']))[p['field'] as keyof S.AllMargins]]),
  margin: ([n, d], p) => {
    const m = S.margin(loop(n, d, p['dt']))
    return tensor([m.gm, m.pm, m.wcg, m.wcp])
  },
  phaseCrossover: ([n, d], p) => {
    const r = S.phaseCrossoverFrequencies(loop(n, d, p['dt']))
    const order = r.omega.map((_, i) => i).sort((i, j) => (r.omega[i] ?? 0) - (r.omega[j] ?? 0))
    return tensor(order.map((i) => (p['part'] === 'omega' ? r.omega[i] : r.gains[i]) ?? Number.NaN))
  },
}

describe('systems/margins against python-control', () => {
  runFixture('systems/margins', FNS, it)
})
