import { goldenSuite } from '../../testing/suite.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import * as D from './index.ts'

const alphaBars = (s: D.NoiseSchedule, t: readonly number[]): ReturnType<typeof tensor> =>
  tensor(t.map((u) => s.alpha(u) ** 2))

/** The numeric parameters of a case that are present. */
const pick = (p: Readonly<Record<string, unknown>>, keys: readonly string[]): Record<string, number> =>
  Object.fromEntries(keys.filter((k) => typeof p[k] === 'number').map((k) => [k, p[k] as number]))

goldenSuite(
  'diffusion schedules against the formulas of their reference implementations',
  'stochastic/diffusion-models',
  {
    discreteAlphaBar: ([betas]) => {
      const b = toFlat(betas)
      const s = D.discreteSchedule(b)
      return alphaBars(
        s,
        b.map((_, i) => (i + 1) / b.length),
      )
    },
    discreteGrid: ([grid]) => {
      const s = D.discreteSchedule(D.linearBetas(1000))
      return alphaBars(s, toFlat(grid))
    },
    cosineBetas: (_inputs, p) => tensor(D.cosineBetas(p['n'] as number)),
    cosineAlphaBar: ([t]) => alphaBars(D.cosineSchedule(), toFlat(t)),
    sigmoidAlphaBar: ([t], p) => alphaBars(D.sigmoidSchedule(pick(p, ['start', 'end', 'tau'])), toFlat(t)),
    vpAlpha: ([t], p) => {
      const s = D.linearSchedule(pick(p, ['betaMin', 'betaMax']))
      return tensor(toFlat(t).map((u) => s.alpha(u)))
    },
    edmPreconditioning: ([sigma], p) =>
      tensor(
        toFlat(sigma).map((v) => {
          const c = D.edmPreconditioning(v, p['sigmaData'] as number)
          return [c.cSkip, c.cOut, c.cIn, c.cNoise]
        }),
      ),
  },
)
