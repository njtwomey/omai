import { type EvidenceEstimatorEntry, evidenceEstimators } from 'omai-engine/probprog/inference'
import { gammaPoisson, normalNormal } from 'omai-engine/probprog/models'
import type { AnytimeRunSpec } from '../../../shared/useAnytimeRuns.ts'

/** The conjugate models on offer: each has the evidence in closed form. */
export const MODELS = [
  { id: 'normal-normal', label: 'normal mean, 12 points', benchmark: normalNormal },
  { id: 'gamma-poisson', label: 'Poisson rate, 15 counts', benchmark: gammaPoisson },
] as const
export type ModelId = (typeof MODELS)[number]['id']

/** The estimators raced, with the names the page shows. */
export const ESTIMATORS = [
  ['nested', 'nested sampling'],
  ['bridge-sampling', 'bridge sampling'],
  ['stepping-stone', 'stepping stone'],
  ['thermodynamic-integration', 'thermodynamic integration'],
  ['importance-evidence', 'importance sampling'],
  ['tempered-smc', 'tempered SMC'],
] as const

/** What a run reads at a snapshot: the log evidence so far and its standard error, where the estimator has one. */
export interface Estimate {
  readonly value: number
  readonly standardError: number | undefined
}

/** The exact log evidence of a model (a closed form). */
export function exactLogEvidence(model: ModelId): number {
  const truth = MODELS.find((m) => m.id === model)?.benchmark.truth()
  if (truth?.logEvidence === undefined) throw new Error(`no closed-form evidence for ${model}`)
  return truth.logEvidence
}

/** One run per estimator on the chosen model and seed, a snapshot every half second. */
export function evidenceRuns(model: ModelId, seed: number): AnytimeRunSpec<Estimate | undefined>[] {
  const bound = MODELS.find((m) => m.id === model)?.benchmark.bound()
  if (bound === undefined) return []
  return ESTIMATORS.flatMap(([id, label]) => {
    const entry: EvidenceEstimatorEntry | undefined = evidenceEstimators[id]
    if (entry?.make === undefined) return []
    const method = entry.make(bound)
    return [
      {
        id,
        label,
        // the estimators of the table are made for any state and path
        method: method as never,
        seed,
        snapshotMs: 500,
        derive: (s, x): Estimate | undefined => {
          const e = method.posterior?.(x.state as never, s.config as never)?.logEvidence
          return e === undefined || !Number.isFinite(e.value)
            ? undefined
            : { value: e.value, standardError: e.standardError }
        },
      },
    ]
  })
}
