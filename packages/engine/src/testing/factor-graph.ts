// What the tests of the model-to-factor-graph compiler (`lib/probprog/model/factor-graph/`) share: compile a bound model and run EP on
// it (with the checks every such test makes first), read a belief, and look at the graph a mapping made (the factors of a site, the
// operators of a fragment). Node only; not published.
import { expect } from 'vitest'
import {
  type CompiledGraph,
  compileGraph,
  epAlgorithm,
  defaultOperators as factorOperators,
  type MpState,
  marginalsOf,
} from '../lib/message-passing/index.ts'
import type { BoundModel } from '../lib/probprog/model/bind.ts'
import type { FactorGraphCompilation } from '../lib/probprog/model/factor-graph/types.ts'
import { factorGraphVerdict } from '../lib/probprog/model/factor-graph/verdict.ts'
import { toFactorGraph } from '../lib/probprog/model/factors.ts'
import { siteOfId } from '../lib/probprog/model/ids.ts'
import { runAlgorithm } from '../lib/trace/index.ts'

/** What `runEp` does besides compiling: the knobs of the run and which checks it makes on the way. */
export interface RunEpOptions {
  /** The tolerance of the run (default 1e-10). */
  readonly tol?: number
  /** The damping of the messages (default: the engine's). */
  readonly damping?: number
  /** The most sweeps (default 400). */
  readonly maxSteps?: number
  /** Check that the engine's verdict for EP on the graph is ok (the graph runs under EP with no operator missing a rule). */
  readonly verdict?: boolean
  /** Check that the run converged or completed. */
  readonly converged?: boolean
  /** More options of `epAlgorithm`. */
  readonly algorithm?: Readonly<Record<string, unknown>>
  /** Compile with the scalar normals that an observation reads together packed (`toFactorGraph`'s `packScalars`; default: not). */
  readonly packScalars?: boolean
  /** Compile with each component's mean and precision fused into one normal-gamma variable where the model writes the conjugate pair (`conjugatePairs`; default: not). */
  readonly conjugatePairs?: boolean
}

/** The compiled model and the state EP reached. */
export interface EpRun {
  readonly c: FactorGraphCompilation
  readonly g: CompiledGraph
  readonly r: { readonly state: unknown; readonly status: string | undefined }
}

/**
 * Compile a bound model to a factor graph (checking that nothing was refused), compile the graph for the engine and run expectation
 * propagation on it. The checks every compiler test makes first are here: no issues, and with `verdict` that EP can run the graph.
 */
export function runEp(bound: BoundModel, options: RunEpOptions = {}): EpRun {
  const c = toFactorGraph(bound, {
    ...(options.packScalars === undefined ? {} : { packScalars: options.packScalars }),
    ...(options.conjugatePairs === undefined ? {} : { conjugatePairs: options.conjugatePairs }),
  })
  expect(c.issues).toEqual([])
  const g = compileGraph(c.graph as never, factorOperators())
  if (options.verdict === true) expect(factorGraphVerdict(c, g, 'ep').ok).toBe(true)
  const r = runAlgorithm(
    epAlgorithm(g, {
      tol: options.tol ?? 1e-10,
      ...(options.damping === undefined ? {} : { damping: options.damping }),
      ...options.algorithm,
    }),
    { maxSteps: options.maxSteps ?? 400 },
  )
  if (options.converged === true) expect(['converged', 'completed']).toContain(r.status)
  return { c, g, r }
}

/** The state of a run, typed. */
export const stateOf = (run: EpRun): MpState => run.r.state as MpState

/** The marginals of a run, by variable name. */
export const marginalsOfRun = (run: EpRun): Record<string, Record<string, unknown>> =>
  marginalsOf(run.g, stateOf(run)) as Record<string, Record<string, unknown>>

/** The mean and standard deviation of a scalar Gaussian belief. */
export function belief(marginals: Record<string, Record<string, unknown>>, name: string): { mean: number; sd: number } {
  const m = marginals[name] as Record<string, number>
  return { mean: m['mean'] as number, sd: Math.sqrt(m['variance'] as number) }
}

/** The factors of a compiled graph that belong to a site (an element, or an auxiliary variable of the site, counts), with their ids. */
export function factorsOf(
  c: FactorGraphCompilation,
  site: string,
): { readonly id: string; readonly operator: string }[] {
  return (c.graph?.factors ?? [])
    .filter((f) => siteOfId(f.id ?? '') === `factor:${site}`)
    .map((f) => ({ id: f.id ?? '', operator: f.operator }))
}

/** The operator names of the factors of a site, in order (the fragment a mapping made). */
export const fragmentOf = (c: FactorGraphCompilation, site: string): string[] =>
  factorsOf(c, site).map((f) => f.operator)

/** The operator names of every factor of a compiled graph, in order. */
export const operatorsOf = (c: FactorGraphCompilation): string[] => (c.graph?.factors ?? []).map((f) => f.operator)
