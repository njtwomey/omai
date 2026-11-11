/**
 * Reading a run back in the spec's shapes: `factorGraphPosterior` builds the library's `Posterior`, keyed by site name, from
 * a message-passing state and the compilation's site layouts (one reader per message family: Gaussian, Gaussian vector,
 * gamma, normal-gamma, beta, Bernoulli, discrete, Dirichlet, Wishart), and `graphOnSpec` maps the ids of a compiled graph
 * back to the spec's nodes.
 */

import { type Json, OmaiError } from '../../../base/index.ts'
import { transformed } from '../../../dists/compose/index.ts'
import {
  bernoulli,
  beta,
  bijectors,
  categorical,
  type Distribution,
  dirichlet,
  gamma,
  inverseGamma,
  inverseWishart,
  multivariateNormal,
  normal,
  studentT,
  wishart,
} from '../../../dists/index.ts'
import {
  type CompiledGraph,
  evidenceDefects,
  evidenceOf,
  type FactorGraphSpec,
  type MpState,
  marginalsOf,
  projectionOf,
} from '../../../message-passing/index.ts'
import { inv } from '../../../numerics/linalg/index.ts'
import { asTensor, fromStorage, type Value } from '../../../tensor/index.ts'
import { definePosterior, type Posterior } from '../../posterior.ts'
import { siteOfId } from '../ids.ts'
import type { FactorGraphCompilation } from './types.ts'
import type { MessageRule } from './verdict.ts'

const num = (v: Json | undefined): number => (typeof v === 'number' ? v : Number.NaN)
const nums = (v: Json | undefined): number[] => (Array.isArray(v) ? (v as number[]) : [])

/**
 * The library's posterior of a message-passing state, keyed by site name: for each latent and recorded derived site of the
 * spec, the marginal as a distribution of the spec's shape (`normal` with the dims' batch shape, a `multivariateNormal`
 * for a vector site the compiler kept packed, `gamma`, `beta`, `bernoulli`, `categorical`, `dirichlet`). Variables with no
 * density yet (a state early in a run) read as an unbounded spread around zero and are listed in
 * `diagnostics.improper`; the diagnostics carry the sweeps, the last change, the improper-message counts and the
 * evidence or bound.
 *
 * @param c - The compilation (it must have a graph) and the compiled graph.
 * @param g - The graph compiled from `c.graph`.
 * @param state - A state of the run.
 * @param options - The method's name, the message rule (for the evidence), the status.
 * @returns The posterior (form `variational`).
 * @throws OmaiError with code `value` when the compilation has no graph.
 * @concept model-spec
 * @index function The Posterior of a message-passing state, by site name, in the spec's shapes.
 * @example
 * ```ts
 * import { compileGraph } from 'omai-engine/message-passing'
 * import { defineModel, factorGraphPosterior, toFactorGraph } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * import { defaultOperators, epAlgorithm } from 'omai-engine/message-passing'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const m = defineModel('m', (b) => { b.observe('y', normal(b.latent('mu', normal(0, 2)), 1), b.data('y', { dims: ['row'] })) })
 * const c = toFactorGraph(m.bind({ y: [1, 2, 3] }))
 * const g = compileGraph(c.graph as never, defaultOperators())
 * const r = runAlgorithm(epAlgorithm(g), { maxSteps: 50 })
 * Number(factorGraphPosterior(c, g, r.state).mean('mu').data[0]).toFixed(4) // => '1.8462'
 * ```
 */
export function factorGraphPosterior(
  c: FactorGraphCompilation,
  g: CompiledGraph,
  state: MpState,
  options: {
    readonly engine?: string
    readonly rule?: MessageRule
    readonly status?: 'completed' | 'converged' | 'stopped'
    /** Why the run did not converge, when it did not (see `mpVerdict`): carried in the diagnostics. */
    readonly reason?: string
    /** An evidence to report where the state's own is undefined (an improper cavity of a plated factor), and the method that gave it. */
    readonly evidence?: { readonly value: number; readonly method: string }
  } = {},
): Posterior {
  if (c.graph === undefined)
    throw new OmaiError('value', `factor graph of '${c.name}' has issues: ${c.issues.map((i) => i.reason).join('; ')}`)
  const rule = options.rule ?? 'ep'
  const marginals = marginalsOf(g, state)
  const out: Record<string, Distribution> = {}
  const improper: string[] = []
  const sites: string[] = []
  const T = (xs: readonly number[], shape: readonly number[]): Value =>
    fromStorage(Float64Array.from(xs), shape, 'float64')
  for (const s of c.sites) {
    if (!s.posterior) continue
    sites.push(s.site)
    const m = (name: string | null): Record<string, Json> => marginals[name ?? ''] ?? {}
    const shape = s.shape
    if (s.layout === 'packed') {
      const mm = m(s.site)
      const mean = nums(mm['mean'])
      const cov = (mm['covariance'] as number[][] | undefined) ?? []
      if (mean.length === 0) {
        improper.push(s.site)
        const d = s.variables.length > 0 ? (g.variables[g.index.get(s.site) as number]?.family.dim ?? 1) : 1
        out[s.site] = multivariateNormal(new Array<number>(d).fill(0), {
          covariance: Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? 1e12 : 0))),
        })
      } else out[s.site] = multivariateNormal(mean, { covariance: cov })
      continue
    }
    const vs = s.variables.map((v) => m(v))
    const bad = vs.some((x) => x['improper'] === true)
    if (bad) improper.push(s.site)
    if (s.family === 'gaussian-mv') {
      // one Gaussian vector per index of the plate dimensions: the batch of their marginals
      const d = shape[shape.length - 1] as number
      const means: number[] = []
      const covs: number[] = []
      for (const x of vs) {
        const mean = nums(x['mean'])
        const cov = (x['covariance'] as number[][] | undefined) ?? []
        for (let i = 0; i < d; i++) {
          means.push(mean[i] ?? 0)
          for (let j = 0; j < d; j++) covs.push(cov[i]?.[j] ?? (i === j ? 1e12 : 0))
        }
      }
      out[s.site] = multivariateNormal(T(means, shape), { covariance: T(covs, [...shape, d]) })
      continue
    }
    switch (s.family) {
      case 'wishart': {
        // The matrix site: a Wishart of the precision, or the inverse Wishart of the covariance (the matrix's inverse).
        const x = vs[0] ?? {}
        const proper = x['improper'] !== true
        const scale = (x['scale'] as number[][] | undefined) ?? []
        const d =
          scale.length > 0
            ? scale.length
            : (g.variables[g.index.get(s.variables[0] as string) as number]?.family.dim ?? 1)
        const nu = proper ? num(x['nu']) : d + 1
        const matrix = proper
          ? scale
          : Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? 1e6 : 0)))
        out[s.site] =
          s.reciprocal === true
            ? inverseWishart(nu, inv(asTensor(matrix, 'float64')))
            : wishart(nu, { covariance: asTensor(matrix, 'float64') })
        break
      }
      case 'gaussian':
        out[s.site] = normal(
          T(
            vs.map((x) => (x['improper'] === true ? 0 : num(x['mean']))),
            shape,
          ),
          T(
            vs.map((x) => (x['improper'] === true ? 1e6 : Math.sqrt(num(x['variance'])))),
            shape,
          ),
        )
        break
      case 'gamma': {
        const shapes = T(
          vs.map((x) => (x['improper'] === true ? 1 : num(x['shape']))),
          shape,
        )
        const rates = T(
          vs.map((x) => (x['improper'] === true ? 1e-6 : num(x['rate']))),
          shape,
        )
        out[s.site] = s.reciprocal === true ? inverseGamma(shapes, rates) : gamma(shapes, rates)
        break
      }
      case 'normal-gamma': {
        // The joint of a mean and a precision shares one variable: the mean's marginal is a Student t (2a degrees of freedom around m,
        // scale sqrt(b / (a beta))), the precision's a gamma (an inverse gamma for a variance).
        const x = vs[0] ?? {}
        const proper = x['improper'] !== true
        const [m, beta, a, b] = [num(x['m']), num(x['beta']), num(x['a']), num(x['b'])]
        if (s.part === 'mean') out[s.site] = proper ? studentT(2 * a, m, Math.sqrt(b / (a * beta))) : normal(0, 1e6)
        else {
          const [sh, rt] = proper ? [a, b] : [1, 1e-6]
          out[s.site] = s.reciprocal === true ? inverseGamma(sh, rt) : gamma(sh, rt)
        }
        break
      }
      case 'beta':
        if (s.correlation === true) {
          const x = vs[0] ?? {}
          const proper = x['improper'] !== true
          out[s.site] = transformed(beta(proper ? num(x['a']) : 1, proper ? num(x['b']) : 1), bijectors.affine(-1, 2))
          break
        }
        out[s.site] = beta(
          T(
            vs.map((x) => (x['improper'] === true ? 1 : num(x['a']))),
            shape,
          ),
          T(
            vs.map((x) => (x['improper'] === true ? 1 : num(x['b']))),
            shape,
          ),
        )
        break
      case 'bernoulli':
        out[s.site] = bernoulli({
          probs: T(
            vs.map((x) => num(x['p'])),
            shape,
          ),
        })
        break
      case 'discrete': {
        const k = nums(vs[0]?.['probs']).length
        out[s.site] = categorical({
          probs: T(
            vs.flatMap((x) => (nums(x['probs']).length === k ? nums(x['probs']) : new Array<number>(k).fill(1 / k))),
            [...shape, k],
          ),
        })
        break
      }
      case 'dirichlet': {
        const k = nums(vs[0]?.['alpha']).length
        out[s.site] = dirichlet(
          T(
            vs.flatMap((x) => nums(x['alpha'])),
            [...shape, k],
          ),
        )
        break
      }
      default:
        break
    }
  }
  const evidence = evidenceOf(g, { projection: projectionOf(rule) }, state)
  return definePosterior({
    form: 'variational',
    engine: options.engine ?? rule,
    sites: sites.filter((s) => out[s] !== undefined),
    status: options.status ?? 'completed',
    marginals: out,
    ...(Number.isFinite(evidence)
      ? { logEvidence: { value: evidence, method: rule === 'vmp' ? 'elbo' : rule === 'bp' ? 'bethe' : 'ep-energy' } }
      : options.evidence !== undefined && Number.isFinite(options.evidence.value)
        ? { logEvidence: options.evidence }
        : {}),
    diagnostics: {
      sweeps: state.sweeps,
      updates: state.updates,
      change: Number.isFinite(state.change) ? state.change : null,
      improper: improper,
      improperMessages: state.improper.events,
      repaired: { skipped: state.improper.skipped, damped: state.improper.damped, reset: state.improper.reset },
      unresolved: state.unresolved ?? 0,
      restarts: state.restarts ?? 0,
      ...(options.reason === undefined ? {} : { reason: options.reason }),
      // Why there is no evidence (when there is none): the factors with no part of it. A plated factor whose cavity is improper
      // (a negative-precision site on the weights) is the usual one; `plates: 'expand'` has a proper cavity per row.
      ...(rule === 'ep' && !Number.isFinite(evidence) && options.evidence === undefined
        ? {
            evidenceUndefinedAt: evidenceDefects(g, { projection: projectionOf(rule) }, state).map(
              (d) => `${d.factor}: ${d.reason}`,
            ),
          }
        : {}),
      variables: g.variables.length,
      factors: g.factors.length,
    },
  })
}

/**
 * Where a graph's ids sit on the spec's picture: each variable id and each factor id of the compiled graph mapped to the id
 * of the spec node or factor site it belongs to (`w[2]` and `y#0[3]` to `w` and `y`, `factor:y[3]` to `factor:y`). A view
 * that draws `toStructure(spec)` uses it to paint the marginals of a run, keyed by variable id, on the spec's nodes.
 *
 * @param graph - The factor graph of a compilation.
 * @returns The two maps.
 * @example
 * ```ts
 * import { graphOnSpec } from 'omai-engine/probprog/model'
 * const g = graphOnSpec({ variables: [{ name: 'y#0[3]', family: { kind: 'gaussian' } }], factors: [{ operator: 'prior', args: ['y#0[3]'], params: { eta: [0, 1] }, id: 'factor:y#0[3]' }] })
 * [g.variables['y#0[3]'], g.factors['factor:y#0[3]']] // => ['y', 'factor:y']
 * ```
 */
export function graphOnSpec(graph: FactorGraphSpec): {
  readonly variables: Readonly<Record<string, string>>
  readonly factors: Readonly<Record<string, string>>
} {
  const variables: Record<string, string> = {}
  for (const v of graph.variables) variables[v.name] = siteOfId(v.name)
  const factors: Record<string, string> = {}
  for (const f of graph.factors) if (f.id !== undefined) factors[f.id] = siteOfId(f.id)
  return { variables, factors }
}
