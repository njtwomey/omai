import { type ModelRun, modelRunFromTape, type PosteriorComparison, posteriorComparison } from 'omai-engine/exposition'
import type { Posterior } from 'omai-engine/probprog'
import { epSpecMethod, laplaceMethod, nutsMethod, type Snapshot } from 'omai-engine/probprog/inference'
import { toStructure } from 'omai-engine/probprog/model'
import { gridTruth, probitRegression } from 'omai-engine/probprog/models'
import { agree, fromPosterior, type MarginalOracle } from 'omai-engine/probprog/validation'
import { key } from 'omai-engine/random'
import { modelGraph } from 'omai-render/structures'
import type { AnytimeRunSpec } from '../../shared/useAnytimeRuns.ts'

/** The model the three engines share: the benchmark's probit regression on its own data (50 rows, 3 weights). */
export const BOUND = probitRegression.bound()
export const COORDINATES = ['w[0]', 'w[1]', 'w[2]'] as const
/** The exact answer: the posterior on a grid of 21 points per axis over the central 10 standard deviations (about two seconds). */
export const exactAnswer = (): ReturnType<typeof gridTruth> =>
  gridTruth(probitRegression.program(), { points: 21, maxRefinements: 0, width: 5 })
export const PICTURE = modelGraph(toStructure(BOUND))

/** What a run keeps of its state at a snapshot. */
export interface Kept {
  /** The posterior the state stands for (empty `sites` while a sampler is still warming up). */
  readonly posterior: Posterior
  /** EP's tape as a model run: beliefs and the messages that passed (only for EP). */
  readonly record?: ModelRun
}

/** The posterior a state of one of the methods stands for. */
const read = (
  method: { posterior?: (state: never, config: never) => Posterior },
  s: { readonly config: unknown },
  x: Snapshot<unknown>,
): Posterior => method.posterior?.(x.state as never, s.config as never) as Posterior

const NUTS = nutsMethod(BOUND)
const EP = epSpecMethod(BOUND)
const LAPLACE = laplaceMethod(BOUND)

/** The three runs: NUTS (as many draws as the budget allows), EP (a few sweeps) and Laplace (an optimiser and a Hessian). */
export function probitRuns(): AnytimeRunSpec<Kept>[] {
  return [
    {
      id: 'nuts',
      label: 'NUTS',
      method: NUTS,
      config: { numChains: 2, numWarmup: 250, numDraws: 1500 },
      derive: (s, x) => ({ posterior: read(NUTS, s, x) }),
    },
    {
      id: 'ep',
      label: 'EP',
      method: EP,
      config: { tol: 1e-10, granularity: 'message' },
      logs: 'all',
      derive: (s, x) => ({
        posterior: read(EP, s, x),
        record: modelRunFromTape(PICTURE, { run: s.run() }),
      }),
    },
    {
      id: 'laplace',
      label: 'Laplace',
      method: LAPLACE,
      derive: (s, x) => ({ posterior: read(LAPLACE, s, x) }),
    },
  ]
}

// A finished run keeps its posterior from one snapshot to the next, and drawing from a Gaussian approximation is not
// cheap (Laplace's takes the best part of half a second), so the draws are made once per posterior.
const drawn = new WeakMap<Posterior, ReturnType<typeof fromPosterior>>()
function approximationOf(name: string, posterior: Posterior): ReturnType<typeof fromPosterior> {
  let a = drawn.get(posterior)
  if (a === undefined) {
    a = fromPosterior(name, posterior, { key: key(0), numDraws: 3000 })
    drawn.set(posterior, a)
  }
  return a
}

/** The methods against the grid: the agreement table and the overlay record of the coordinates. */
export function compare(
  kept: ReadonlyArray<{ readonly name: string; readonly kept: Kept | undefined }>,
  marginals: Readonly<Record<string, MarginalOracle>>,
): PosteriorComparison | undefined {
  const approximations = kept.flatMap((k) =>
    k.kept === undefined || k.kept.posterior.sites.length === 0 ? [] : [approximationOf(k.name, k.kept.posterior)],
  )
  if (approximations.length === 0) return undefined
  const table = agree(approximations, { oracle: marginals, quantities: [...COORDINATES] })
  return posteriorComparison(approximations, {
    truth: marginals,
    table,
    quantities: [...COORDINATES],
    oracle: 'grid',
    quantilePoints: 120,
    gridPoints: 160,
  })
}
