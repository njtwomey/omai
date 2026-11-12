import { type SbcRun, sbcRun } from 'omai-engine/exposition'
import { nutsMethod, sviMethod } from 'omai-engine/probprog/inference'
import { defineModel } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { type SbcMethod, sbcMethod } from 'omai-engine/probprog/validation'
import { foldIn, type Key, key, normal } from 'omai-engine/random'
import { toFlat } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
import type { AnytimeRunSpec } from '../../../shared/useAnytimeRuns.ts'

/** The ranks are among this many draws, so they run from 0 to 19. */
const RANKS = 19
/** The covariate: not centred, so the intercept and the slope are strongly correlated in the posterior. */
const X = [1, 2, 3, 4, 5, 6]
/** The replications each run ends after, if the budget does not end it first. */
export const REPLICATIONS = 60

// region sbc
// y_i = a + b x_i + noise, a and b with N(0, 1) priors: the model the data are drawn from and the one fitted.
const MODEL = defineModel('line', (m) => {
  const n = m.range('row')
  const x = m.data('x', { dims: [n], role: 'covariate' })
  const y = m.data('y', { dims: [n], role: 'response' })
  const a = m.latent('a', d.normal(0, 1))
  const b = m.latent('b', d.normal(0, 1))
  m.observe('y', d.normal(a.add(b.mul(x)), 1), y)
})
// endregion

/** The prior and the model, drawn: the parameters (the truth to rank) and the data they make. */
function simulate(k: Key): { readonly data: number[]; readonly truth: { readonly a: number; readonly b: number } } {
  const [a = 0, b = 0] = toFlat(normal(foldIn(k, 0), [2]))
  const noise = toFlat(normal(foldIn(k, 1), [X.length]))
  return { data: X.map((x, i) => a + b * x + (noise[i] ?? 0)), truth: { a, b } }
}

type Fit = (y: readonly number[], k: Key) => Record<string, number[]>

const boundTo = (y: readonly number[]): ReturnType<typeof MODEL.bind> => MODEL.bind({ x: X, y: [...y] })

/** NUTS: one chain, a short warm-up, 190 draws thinned to the ranks' 19. */
// region sbc:nuts
const nuts: Fit = (y) => {
  const method = nutsMethod(boundTo(y))
  const s = session(method, {
    seed: 1,
    steps: 1_000_000,
    mode: 'eager',
    config: { numChains: 1, numWarmup: 100, numDraws: 190 },
  })
  const p = method.posterior?.(s.state, s.config)
  const a = toFlat(p?.draws?.['a'] as never)
  const b = toFlat(p?.draws?.['b'] as never)
  return { a, b }
}
// endregion

/** A mean-field guide fitted for `steps` steps, drawn from for the ranks. */
// region sbc:guide
const guide =
  (steps: number): Fit =>
  (y, k) => {
    const method = sviMethod(boundTo(y))
    const s = session(method, {
      seed: 1,
      steps,
      mode: 'eager',
      config: { guide: 'autoNormal', learningRate: 0.05, numParticles: 2 },
    })
    const drawn = method.posterior?.(s.state, s.config).sample(k, RANKS) ?? {}
    const a = toFlat(drawn['a'] as never)
    const b = toFlat(drawn['b'] as never)
    return { a, b }
  }
// endregion

/** The three methods under test, with what each is expected to show. */
export const METHODS = [
  { id: 'nuts', label: 'NUTS', fit: nuts, note: 'a consistent sampler: ranks uniform, the curve inside the band' },
  {
    id: 'mean-field',
    label: 'mean-field guide',
    fit: guide(300),
    note: 'the posterior of the intercept and the slope is strongly correlated, a mean-field guide cannot say so and is too narrow: ranks pile up at both ends',
  },
  {
    id: 'early',
    label: 'guide, stopped early',
    fit: guide(25),
    note: 'still near where it started, far from the posterior: biased and miscalibrated',
  },
] as const

/** What a run keeps: the calibration record so far (from the tenth replication). */
export interface Kept {
  readonly record: SbcRun | undefined
  readonly done: number
}

/** One calibration run per method: a replication a step, the record of the ranks so far at each snapshot. */
export function calibrationRuns(): AnytimeRunSpec<Kept>[] {
  return METHODS.map((m) => {
    // region sbc:run
    const method: SbcMethod = sbcMethod({
      name: `sbc-${m.id}`,
      numRanks: RANKS,
      replications: REPLICATIONS,
      key: key(2026),
      nullSimulations: 400,
      simulate,
      fit: (y, k) => m.fit(y, k),
    })
    // endregion
    return {
      id: m.id,
      label: m.label,
      method,
      derive: (_s, x) => {
        const result = method.result(x.state as never)
        return { record: result === undefined ? undefined : sbcRun(result), done: (x.state as { done: number }).done }
      },
    }
  })
}
