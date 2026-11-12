import { type ElboRun, elboRunFromTape } from 'omai-engine/exposition'
import { structuredSviMethod } from 'omai-engine/probprog/inference'
import { banana, bananaDraws } from 'omai-engine/probprog/models'
import { key } from 'omai-engine/random'
import { toFlat } from 'omai-engine/tensor'
import type { AnytimeRunSpec } from '../../../shared/useAnytimeRuns.ts'

/** The guides raced, by the id the page uses, with the knobs that make each. */
export const GUIDES = [
  { id: 'mean-field', label: 'mean-field', config: { guide: 'autoNormal' } },
  { id: 'full-rank', label: 'full-rank', config: { guide: 'autoMultivariateNormal' } },
  { id: 'coupling', label: 'coupling flow', config: { guide: 'autoFlowNormal', flow: 'coupling', numFlows: 4 } },
  { id: 'iaf', label: 'IAF', config: { guide: 'autoIafNormal', numFlows: 4 } },
] as const
export type GuideId = (typeof GUIDES)[number]['id']

/** The banana, a density with no data: x0 ~ N(0, 4), x1 | x0 ~ N(0.5 (x0^2 - 4), 1). */
const BOUND = banana.bound()
const TRUTH = banana.truth().marginals

/** What a run keeps at a snapshot: its record (ELBO over the steps, beliefs, the posterior over the truth) and draws of the guide. */
export interface Kept {
  readonly record: ElboRun
  /** Beliefs so far, kept to build the next record on. */
  readonly snapshots: readonly { readonly step: number; readonly snapshot: unknown }[]
  readonly x0: readonly number[]
  readonly x1: readonly number[]
}

/** Exact draws of the target, for the picture beside the guides'. */
export function targetDraws(n: number): { readonly x0: Float64Array; readonly x1: Float64Array } {
  return bananaDraws(key(11), n)
}

/** One SVI run per guide, with a record of each built at every snapshot from the tape and the beliefs so far. */
export function guideRuns(): AnytimeRunSpec<Kept>[] {
  return GUIDES.map((g) => {
    const method = structuredSviMethod(BOUND)
    return {
      id: g.id,
      label: g.label,
      method,
      config: { ...g.config, learningRate: 0.02, numParticles: 4 },
      seed: 1,
      derive: (s, x, previous) => {
        const posterior = method.posterior?.(x.state as never, s.config as never)
        const snapshots = [
          ...(previous?.snapshots ?? []),
          ...(posterior === undefined || posterior.sites.length === 0
            ? []
            : [{ step: x.t, snapshot: { summary: posterior.summary() } }]),
        ]
        const record = elboRunFromTape(
          { run: s.run(), snapshots },
          { truth: TRUTH, truthName: 'exact', method: g.label },
        )
        const drawn = posterior?.sample(key(3), 500)
        return {
          record,
          snapshots,
          x0: drawn?.['x0'] === undefined ? [] : Array.from(toFlat(drawn['x0'])),
          x1: drawn?.['x1'] === undefined ? [] : Array.from(toFlat(drawn['x1'])),
        }
      },
    }
  })
}
