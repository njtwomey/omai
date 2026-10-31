import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../testing/systems.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import { pade, tf } from './representations.ts'
import { type StepMeasures, stepInfo, stepMeasures } from './responses.ts'

// control.step_info's order.
const asList = (m: StepMeasures): Tensor =>
  tensor([
    m.riseTime,
    m.settlingTime,
    m.settlingMin,
    m.settlingMax,
    m.overshoot,
    m.undershoot,
    m.peak,
    m.peakTime,
    m.steadyStateValue,
  ]) as Tensor

const FNS: Readonly<Record<string, GoldenFn>> = {
  'step-info': ([num, den, t], p) =>
    asList(
      stepInfo(tf(num as Tensor, den as Tensor), {
        t: t as Tensor,
        ...(p['band'] === undefined ? {} : { settlingThreshold: p['band'] as number }),
        ...(p['low'] === undefined ? {} : { riseLimits: [p['low'] as number, p['high'] as number] as const }),
      }),
    ),
  'step-data': ([t, y]) => asList(stepMeasures(toFlat(t as Tensor), toFlat(y as Tensor))),
  'pade-num': (_inputs, p) =>
    pade(p['delay'] as number, p['n'] as number, (p['numdeg'] as number) === -1 ? undefined : (p['numdeg'] as number))
      .num,
  'pade-den': (_inputs, p) =>
    pade(p['delay'] as number, p['n'] as number, (p['numdeg'] as number) === -1 ? undefined : (p['numdeg'] as number))
      .den,
}

describe('systems/measures against python-control', () => {
  runFixture('systems/measures', FNS, it)
})
