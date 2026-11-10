/**
 * The structure report of a spec: what its structure lets a method exploit, as records and as text. It gathers the
 * conjugacy records, the linear-Gaussian blocks, the Markov blankets, the independent components, the width of the
 * discrete part and the minibatch axes, and turns them into plain suggestions that each name the existing method or
 * handler to use and the knob values that would apply (`collapse(...)`, `nuts` with `inferDiscrete`, `svi` on a batch
 * bound with `bind(..., { batch })`, `reparam` with `locScale`). A report is advice for a person to read and act on: the
 * library never applies it, selects a method from it or rewrites a model because of it. `analyseSpec(model).explain()`
 * and `structureReport(model).explain()` print it.
 *
 * @index structure StructureReport: blankets, components, discrete width, conjugate and Gaussian blocks, minibatch axes, suggestions.
 */

import type { Json } from '../../base/index.ts'
import type { SpecAnalysis } from './analysis.ts'
import type { ConjugacyRecord } from './analysis-conjugacy.ts'
import { type LatentMemo, latentsIn, nodesByName } from './analysis-exprs.ts'
import { type GraphStructure, graphStructure } from './analysis-graph.ts'
import type { LinearGaussianBlock } from './analysis-linear-gaussian.ts'
import { canonicalDims } from './ids.ts'
import type { ExprSpec, LatentNode, ModelSpec } from './spec.ts'

/** What a suggestion tells a person to use: an inference method of `omai-engine/probprog/inference` or a handler. */
export interface SuggestedUse {
  readonly kind: 'method' | 'handler'
  /** A key of `methodKinds` (`nuts`, `svi`, `gibbs`, `gaussian`, `conjugate`, `enumeration`) or a handler (`collapse`, `reparam`). */
  readonly name: string
  /** The knob values or options that apply (the names the method or handler takes). */
  readonly knobs?: Readonly<Record<string, Json>>
  /** How to apply it, in a line. */
  readonly how: string
}

/** The kinds of advice a report gives. */
export type SuggestionKind =
  | 'closed-form'
  | 'enumerate'
  | 'collapse'
  | 'conditional'
  | 'gaussian-block'
  | 'split'
  | 'minibatch'
  | 'reparameterise'
  | 'limit'

/** One plain statement about what the structure allows, with the sites it is about and what to use. */
export interface StructureSuggestion {
  readonly kind: SuggestionKind
  readonly sites: readonly string[]
  /** The statement, as a sentence a person can read. */
  readonly text: string
  /** Why, with the exact reasons the analysis found. */
  readonly reason: string
  readonly use: readonly SuggestedUse[]
}

/**
 * What a spec's structure allows, as records and suggestions (see {@link structureReport}).
 *
 * @concept model-analysis
 */
export interface StructureReport extends GraphStructure {
  readonly name: string
  /** The conjugacy records (pairs, hierarchies, joint priors), the exact ones and the near misses with their reasons. */
  readonly conjugateBlocks: readonly ConjugacyRecord[]
  readonly linearGaussianBlocks: readonly LinearGaussianBlock[]
  readonly suggestions: readonly StructureSuggestion[]
  /** The report as text. */
  explain(): string
}

/** What the report reads of the analysis it extends. */
export type BaseAnalysis = Omit<SpecAnalysis, 'explain' | 'structure'>

const list = (xs: readonly string[]): string => xs.map((x) => `\`${x}\``).join(', ')
const set = (xs: readonly string[]): string => `{${xs.join(', ')}}`

function dimsText(spec: ModelSpec, name: string): string {
  const n = spec.nodes.find((x) => x.name === name && x.kind !== 'data')
  const dims = n === undefined ? [] : canonicalDims(spec, n.dims)
  return dims.length === 0 ? '' : ` over ${dims.join(' x ')}`
}

function suggestions(
  spec: ModelSpec,
  a: BaseAnalysis,
  g: GraphStructure,
  sizeOf: (dim: string) => number | undefined,
  scaleLatents: (scale: ExprSpec) => string[],
): StructureSuggestion[] {
  const out: StructureSuggestion[] = []
  const latents = spec.nodes.filter((n): n is LatentNode => n.kind === 'latent')

  // Discrete sites that can be summed out.
  for (const c of g.discretePart.components) {
    const states = c.cliqueStates
    const per =
      states === undefined
        ? 'a number of states that is not known from the spec'
        : `${states} joint state${states === 1 ? '' : 's'} per element`
    const cost = c.elementwise
      ? `${per}${c.elements === undefined ? '' : `, ${c.elements} independent element${c.elements === 1 ? '' : 's'}`}; width ${c.width}`
      : `its elements interact along ${c.coupledAlong.join(', ')}, so the joint has ${states === undefined ? 'exponentially many' : `${states} to the power of the number of elements`} states; width ${c.width}`
    const feasible = c.elementwise && (states === undefined || states <= 1024)
    out.push({
      kind: 'enumerate',
      sites: c.sites,
      text: feasible
        ? `site${c.sites.length > 1 ? 's' : ''} ${list(c.sites)} (finite discrete${c.sites.map((s) => dimsText(spec, s)).join('')}) can be summed out: NUTS and VI can enumerate ${c.sites.length > 1 ? 'them' : 'it'} (${cost})`
        : `site${c.sites.length > 1 ? 's' : ''} ${list(c.sites)} (finite discrete) cannot be summed out cheaply (${cost}): sample ${c.sites.length > 1 ? 'them' : 'it'} with Gibbs instead`,
      reason: `the discrete part has ${c.sites.length} site${c.sites.length === 1 ? '' : 's'} in this component, elimination order ${set(c.order)} gives induced width ${c.width}`,
      use: feasible
        ? [
            {
              kind: 'method',
              name: 'nuts',
              knobs: { inferDiscrete: true },
              how: 'NUTS sums the discrete sites out of the density; `inferDiscrete: true` draws them back for each draw',
            },
            ...(a.continuous.length === 0
              ? [
                  {
                    kind: 'method' as const,
                    name: 'enumeration',
                    how: 'every latent is finite discrete: the exact posterior by enumeration',
                  },
                ]
              : []),
          ]
        : [{ kind: 'method', name: 'gibbs', how: 'exact conditionals of the discrete sites, one sweep per step' }],
    })
  }

  // Closed form for the whole model.
  const closed =
    a.latent.length > 0 &&
    a.latent.every((l) => a.conjugate.some((c) => c.prior === l && c.known && c.hyperparametersKnown && c.exclusive))
  if (closed)
    out.push({
      kind: 'closed-form',
      sites: a.latent,
      text: `every latent (${list(a.latent)}) is conjugate to its observations with known hyperparameters: the posterior is closed form`,
      reason:
        'each latent has a conjugate pair with known fixed parameters, known hyperparameters and nothing else depending on it',
      use: [{ kind: 'method', name: 'conjugate', how: 'closed-form posterior and evidence in one step' }],
    })

  // Collapsing conjugate sites, or the closed-form conditional that collapse cannot take.
  for (const r of a.conjugacy) {
    if (!r.holds || r.kind === 'joint-prior') continue
    const pairs = r.pairs.join(', ')
    const given = r.given.length > 0 ? ` given ${list(r.given)}` : ''
    if (r.collapse.applies) {
      const rest = a.continuous.filter((c) => !r.sites.includes(c))
      out.push({
        kind: 'collapse',
        sites: r.sites,
        text: `site ${list(r.sites)} is conjugate to ${list(r.links.map((l) => l.node))} (${pairs})${given}: collapse can integrate it out and a sampler then moves in ${rest.length === 0 ? 'no continuous latents' : `${list(rest)} alone`}`,
        reason: r.reason,
        use: [
          {
            kind: 'handler',
            name: 'collapse',
            knobs: { sites: [...r.sites] },
            how: `collapsePlan(bound) lists the sites, applyCollapse(bound, plan) gives the collapsed program; recover draws ${r.sites.join(', ')} back from its closed-form posterior`,
          },
        ],
      })
    } else
      out.push({
        kind: 'conditional',
        sites: r.sites,
        text: `site ${list(r.sites)} has a closed-form conditional (${pairs}${given})${r.links.some((l) => l.nodeKind === 'latent') ? ', with latent children' : ''}: a Gibbs step can draw it exactly; the collapse handler does not take it (${r.collapse.reason})`,
        reason: r.reason,
        use: [{ kind: 'method', name: 'gibbs', how: 'conjugate conditionals are drawn exactly inside a Gibbs sweep' }],
      })
  }

  // Gaussian blocks.
  for (const b of a.linearGaussianBlocks) {
    out.push({
      kind: 'gaussian-block',
      sites: b.sites,
      text: b.whole
        ? `the whole model is linear-Gaussian (${list(b.sites)}${b.dimension === undefined ? '' : `, ${b.dimension} variable${b.dimension === 1 ? '' : 's'}`}): the posterior is a Gaussian, exact`
        : `sites ${set(b.sites)} form a Gaussian block conditional on ${set(b.given)}${b.dimension === undefined ? '' : ` (${b.dimension} variable${b.dimension === 1 ? '' : 's'})`}: given those, the block's posterior is Gaussian and its marginal likelihood closed form`,
      reason: b.reason,
      use: b.whole
        ? [
            {
              kind: 'method',
              name: 'gaussian',
              how: 'exact Gaussian posterior (joint solve, Kalman or belief propagation)',
            },
          ]
        : [
            {
              kind: 'method',
              name: 'nuts',
              how: `run NUTS on ${set(b.given)} alone with the block marginalised: no handler integrates a multi-site Gaussian block today (collapse takes one conjugate site at a time), so this is a gap the report names`,
            },
          ],
    })
  }

  // Independent sub-problems.
  const withLatents = g.components.filter((c) => c.latents.length > 0)
  if (withLatents.length > 1)
    out.push({
      kind: 'split',
      sites: withLatents.flatMap((c) => c.latents),
      text: `the model splits into ${withLatents.length} independent sub-problems (${withLatents.map((c) => set(c.latents)).join(', ')}): each can be inferred on its own`,
      reason: 'no density term connects them: the posterior is a product, and the costs add instead of multiplying',
      use: [
        {
          kind: 'method',
          name: 'nuts',
          how: 'one model per component (conditionSpec or a model with only that component), one run each',
        },
      ],
    })

  // Minibatches.
  for (const ax of g.minibatchAxes) {
    const n = sizeOf(ax.dim)
    if (n !== undefined && n < 64) continue
    out.push({
      kind: 'minibatch',
      sites: [...ax.observed],
      text: `dimension \`${ax.dim}\`${n === undefined ? '' : ` (${n} elements)`} is exchangeable (nothing reduces over it or looks it up by index): a minibatch of it gives an unbiased density estimate${ax.local.length > 0 ? `, with ${list(ax.local)} local to each element` : ''}`,
      reason: `${list(ax.observed)} are over ${ax.dim} and no expression reduces over it`,
      use: [
        {
          kind: 'method',
          name: 'svi',
          knobs: {},
          how: `bind the model with \`{ batch: { dim: '${ax.dim}', size: ${n === undefined ? 64 : Math.min(64, Math.max(8, Math.floor(n / 4)))} } }\`: log densities over ${ax.dim} are rescaled by size over batch size`,
        },
      ],
    })
  }

  // Funnel geometry: a plate of latents whose scale is a latent.
  for (const x of latents) {
    const scale = x.dist.params['scale']
    if (scale === undefined || x.dims.length === 0) continue
    const family = x.dist.family
    if (!['normal', 'cauchy', 'studentT', 'laplace'].includes(family)) continue
    const ls = scaleLatents(scale)
    if (ls.length === 0) continue
    out.push({
      kind: 'reparameterise',
      sites: [x.name],
      text: `site \`${x.name}\` is centred (its scale is the latent ${list(ls)}): the funnel geometry slows NUTS when the data for each element are weak; sampling the standardised value avoids it`,
      reason: `${x.name} is ${family} over ${x.dims.join(', ')} with scale depending on ${ls.join(', ')}`,
      use: [
        {
          kind: 'handler',
          name: 'reparam',
          knobs: { [x.name]: 'locScale(0)' },
          how: `reparam(program, { ${x.name}: locScale(0) }) makes the program sample a standard value and transform it`,
        },
      ],
    })
  }

  // What cannot be used.
  if (a.opaque.length > 0)
    out.push({
      kind: 'limit',
      sites: a.opaque,
      text: `nodes ${list(a.opaque)} are opaque (a custom function or a family with no exponential-family form): message passing and the closed-form methods do not apply, the sampling methods do`,
      reason: 'message passing needs every factor to be a registered operator and an exponential-family distribution',
      use: [{ kind: 'method', name: 'nuts', how: 'sampling methods run on any differentiable density' }],
    })
  return out
}

/** The report as text: what was found, then the suggestions. */
export function explainReport(a: BaseAnalysis, r: Omit<StructureReport, 'explain'>): string {
  const lines: string[] = []
  const kinds = [
    a.continuous.length > 0 ? `${a.continuous.length} continuous latent${a.continuous.length === 1 ? '' : 's'}` : '',
    a.finiteDiscrete.length > 0 ? `${a.finiteDiscrete.length} finite discrete` : '',
    a.unboundedDiscrete.length > 0 ? `${a.unboundedDiscrete.length} unbounded discrete` : '',
    `${a.observed.length} observed`,
  ].filter((x) => x !== '')
  lines.push(`Structure of ${r.name}: ${kinds.join(', ')}; plates ${a.dims.map((d) => d.name).join(', ') || 'none'}.`)
  const comps = r.components.filter((c) => c.latents.length > 0)
  if (comps.length > 0)
    lines.push(
      `Independent components (${comps.length}): ${comps.map((c) => `${set(c.latents)} with ${c.observed.length > 0 ? list(c.observed) : 'no observations'}`).join('; ')}.`,
    )
  if (r.conjugateBlocks.length > 0) {
    lines.push('Conjugacy:')
    for (const c of r.conjugateBlocks)
      lines.push(
        `  - ${c.sites.join(', ')} [${c.kind}${c.holds ? '' : ', not exact'}]: ${c.reason}. Posterior: ${c.posterior}. Collapse handler: ${c.collapse.applies ? 'yes' : `no (${c.collapse.reason})`}.`,
      )
  } else lines.push('Conjugacy: none found.')
  if (r.linearGaussianBlocks.length > 0) {
    lines.push('Linear-Gaussian blocks:')
    for (const b of r.linearGaussianBlocks)
      lines.push(
        `  - ${set(b.sites)}${b.given.length > 0 ? ` given ${set(b.given)}` : ' (the whole model)'}, explaining ${list(b.observed)}: ${b.reason}.`,
      )
  }
  const d = r.discretePart
  if (d.sites.length > 0)
    lines.push(`Discrete part: ${list(d.sites)}; induced width ${d.width}; elimination order ${set(d.orderHint)}.`)
  if (r.minibatchAxes.length > 0)
    lines.push(
      `Minibatch axes: ${r.minibatchAxes.map((m) => `${m.dim}${m.size === undefined ? '' : ` (${m.size})`}`).join(', ')}.`,
    )
  const blank = r.blankets.filter((b) => b.blanket.length > 0)
  if (blank.length > 0)
    lines.push(
      `Markov blankets: ${blank.map((b) => `${b.site}: ${set(b.blanket)}${b.perElement.length > 0 ? ` (per element of ${b.perElement.join(', ')})` : ''}`).join('; ')}.`,
    )
  if (r.suggestions.length === 0)
    lines.push('Suggestions: none; nothing in the structure is specific enough to exploit.')
  else {
    lines.push('Suggestions (advice, nothing is applied):')
    for (const s of r.suggestions) {
      lines.push(`  - ${s.text}.`)
      lines.push(`      why: ${s.reason}`)
      for (const u of s.use)
        lines.push(
          `      use ${u.kind} \`${u.name}\`${u.knobs === undefined ? '' : ` ${JSON.stringify(u.knobs)}`}: ${u.how}`,
        )
    }
  }
  return lines.join('\n')
}

/**
 * Build the structure report from a spec and its analysis (the function behind `analyseSpec(model).structure()` and
 * `structureReport(model)`).
 *
 * @param spec - The spec.
 * @param a - Its analysis without the report's own methods.
 * @param sizeOf - The size of a named dimension, where known.
 * @returns The report.
 */
export function buildStructureReport(
  spec: ModelSpec,
  a: BaseAnalysis,
  sizeOf: (dim: string) => number | undefined,
): StructureReport {
  const byName = nodesByName(spec)
  const memo: LatentMemo = new Map()
  const g = graphStructure({ spec, byName, memo, sizeOf })
  const data = {
    name: a.name,
    ...g,
    conjugateBlocks: a.conjugacy,
    linearGaussianBlocks: a.linearGaussianBlocks,
    suggestions: suggestions(spec, a, g, sizeOf, (e) => [...latentsIn(e, byName, memo)]),
  }
  const report = { ...data } as unknown as StructureReport
  Object.defineProperty(report, 'explain', { value: () => explainReport(a, data), enumerable: false })
  return report
}
