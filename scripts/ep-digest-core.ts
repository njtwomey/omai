/**
 * The machinery of the EP digest (`scripts/ep-digest.ts`, `make ep-digest`, `make ep-digest-check`): the proof that a
 * behaviour-preserving refactoring of the EP capability changed nothing. For every EP atom, every benchmark model and every
 * EP program the library can run without the pocs it computes the compiled factor graph (as canonical JSON, with its
 * sha256), the verdict of the run, its beliefs and evidence, the atom's check rows and the variational (vmp) run; and the
 * registry (the operators and what each declares, the knob spaces). `compare` says at which level two digests agree:
 * bit-identical, exact (graphs, verdicts, counts identical, numbers equal to 12 significant digits) or within a stated
 * tolerance. Nothing here reads a clock or a random source: a digest of unchanged code is byte-identical to the last.
 */

import { createHash } from 'node:crypto'
import {
  bpDefaults,
  compileGraph,
  defaultOperators,
  epDefaults,
  type FactorOperator,
  messagePassingKnobs,
  vmpDefaults,
} from '../packages/engine/src/lib/message-passing/index.ts'
import {
  atomBeliefs,
  atomBound,
  atomCheck,
  type atomReference,
  atomVerdict,
  type EpAtom,
  epAtoms,
  epCoverage,
  verifyAtom,
  vmpCoverage,
} from '../packages/engine/src/lib/probprog/ep/index.ts'
import {
  epGridKnobs,
  epGridMethod,
  epMultiStartKnobs,
  epSpecMethod,
  vmpSpecMethod,
} from '../packages/engine/src/lib/probprog/inference/index.ts'
import { analysisOf } from '../packages/engine/src/lib/probprog/inference/model-input.ts'
import type { BoundModel } from '../packages/engine/src/lib/probprog/model/index.ts'
import { analyseSpec, toFactorGraph } from '../packages/engine/src/lib/probprog/model/index.ts'
import { benchmarkModels } from '../packages/engine/src/lib/probprog/models/index.ts'
import type { Posterior } from '../packages/engine/src/lib/probprog/posterior.ts'
import { toFlat } from '../packages/engine/src/lib/tensor/index.ts'
import type { Config } from '../packages/engine/src/lib/trace/index.ts'
import { session } from '../packages/engine/src/lib/trace/index.ts'

// ----- canonical JSON, hashing, rounding -----

/** A JSON value as the digest keeps it. */
export type J = null | boolean | number | string | readonly J[] | { readonly [k: string]: J }

/** The text of a number at full precision (the shortest string that reads back to the same double); the non-finite ones are named. */
const numText = (n: number): string =>
  Number.isNaN(n)
    ? '"NaN"'
    : n === Number.POSITIVE_INFINITY
      ? '"Infinity"'
      : n === Number.NEGATIVE_INFINITY
        ? '"-Infinity"'
        : Object.is(n, -0)
          ? '-0'
          : String(n)

/**
 * Canonical JSON: keys sorted, no whitespace, numbers at full precision, `undefined` members dropped, typed arrays as arrays,
 * a function as the word `[function]` (a space carries a `when` or a guard; a graph never should).
 *
 * @param v - The value.
 * @returns The text.
 */
export function canonical(v: unknown): string {
  if (v === undefined) return 'null'
  if (v === null) return 'null'
  if (typeof v === 'number') return numText(v)
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (typeof v === 'string') return JSON.stringify(v)
  if (typeof v === 'bigint') return JSON.stringify(String(v))
  if (typeof v === 'function') return '"[function]"'
  if (ArrayBuffer.isView(v)) return canonical(Array.from(v as unknown as ArrayLike<number>))
  if (Array.isArray(v)) return `[${v.map(canonical).join(',')}]`
  if (v instanceof Map) return canonical(Object.fromEntries([...v.entries()].map(([k, x]) => [String(k), x])))
  if (v instanceof Set) return canonical([...v])
  const o = v as Record<string, unknown>
  const keys = Object.keys(o)
    .filter((k) => o[k] !== undefined)
    .sort()
  return `{${keys.map((k) => `${JSON.stringify(k)}:${canonical(o[k])}`).join(',')}}`
}

/** sha256 of a text, in hex. */
export const sha256 = (s: string): string => createHash('sha256').update(s).digest('hex')

/** A canonical value read back as a plain value (the non-finite names stay strings). */
export const plain = (v: unknown): J => JSON.parse(canonical(v)) as J

/** A number rounded to 12 significant digits (what the committed digest keeps). */
export function round12(n: number): number {
  return Number.isFinite(n) && n !== 0 ? Number(n.toPrecision(12)) : n
}

/** A value with every number rounded to 12 significant digits, except the log evidences (the tolerance on those is 1e-12 absolute, finer than 12 digits of a value of tens). */
export function rounded(v: J, key = ''): J {
  if (typeof v === 'number') return key === 'evidence' ? v : round12(v)
  if (Array.isArray(v)) return v.map((x) => rounded(x, key))
  if (v !== null && typeof v === 'object')
    return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, rounded(x as J, k)]))
  return v
}

// ----- what an item holds -----

/**
 * One item of the digest: an atom, a benchmark model, a variational run or a program.
 *
 * `exact` is compared as text (graph hashes, verdicts, counts, reasons, pass flags); `num` is compared to a tolerance (beliefs,
 * evidence, check statistics); `bits` is the hash of the full-precision `num` section, so a run that changed a last digit is seen.
 */
export interface Item {
  readonly kind: 'atom' | 'model' | 'vmp' | 'program'
  readonly exact: J
  readonly num: J
  readonly bits: string
}

/** The digest of a run of the whole set (or a chosen part of it). */
export interface Digest {
  readonly format: 1
  readonly items: { readonly [key: string]: Item }
  /** The registry: the operators and what each declares, the knob spaces. */
  readonly registry: J
}

/** The graphs of the items, by key and variant, as canonical JSON text (kept in a cache, not in the committed digest). */
export type GraphStore = Map<string, string>

const KEYS_DROPPED = new Set(['label', 'description', 'descriptions', 'unit'])

/** A space (or any value) as data with the descriptive text left out: the structure, bounds, choices and defaults a run depends on. */
function structural(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(structural)
  if (v !== null && typeof v === 'object' && !ArrayBuffer.isView(v) && !(v instanceof Map) && !(v instanceof Set))
    return Object.fromEntries(
      Object.entries(v as Record<string, unknown>)
        .filter(([k]) => !KEYS_DROPPED.has(k))
        .map(([k, x]) => [k, structural(x)]),
    )
  return v
}

// ----- the compiled factor graph -----

/** The three ways the engine compiles a model: EP's (plates vectorised, scalars packed), the rows expanded, and the variational one (scalars not packed). */
export const VARIANTS: readonly {
  readonly name: string
  readonly plates: 'vectorise' | 'expand'
  readonly packScalars: boolean
}[] = [
  { name: 'ep', plates: 'vectorise', packScalars: true },
  { name: 'rows', plates: 'expand', packScalars: true },
  { name: 'vmp', plates: 'vectorise', packScalars: false },
]

/** What a compiled graph summarises to for the digest: its hash and its size. The full JSON goes to the store. */
export function graphOf(key: string, bound: BoundModel, v: (typeof VARIANTS)[number], store: GraphStore): J {
  let text: string
  let sizes: J
  try {
    const c = toFactorGraph(bound, { analysis: analyseSpec(bound), plates: v.plates, packScalars: v.packScalars })
    let compiledFailure: string | null = null
    let compiled: J = null
    if (c.graph !== undefined) {
      try {
        const g = compileGraph(c.graph, defaultOperators())
        // What the engine reads from the graph: each variable's family and each factor's operator with its resolved ports.
        compiled = {
          variables: g.variables.map((x) => ({
            name: x.name,
            family: x.family.kind,
            size: x.family.dim,
            edges: x.edges,
            definer: x.definer ?? null,
          })),
          factors: g.factors.map((f) => ({
            id: f.id,
            operator: f.operator.name,
            power: f.power,
            ports: f.ports.map((p) => ({ variable: p.variable ?? null, family: p.family.kind, edge: p.edge })),
            start: f.start,
          })),
          edges: g.edges,
          hasDefined: g.hasDefined,
        } as unknown as J
      } catch (e) {
        compiledFailure = e instanceof Error ? e.message : String(e)
      }
    }
    text = canonical({ compilation: c, compiled, compiledFailure })
    sizes = {
      variables: c.graph?.variables.length ?? null,
      factors: c.graph?.factors.length ?? null,
      issues: c.issues.length,
      failure: compiledFailure,
    }
  } catch (e) {
    text = canonical({ throws: e instanceof Error ? e.message : String(e) })
    sizes = { throws: e instanceof Error ? e.message : String(e) }
  }
  store.set(`${key}.${v.name}`, text)
  return { hash: sha256(text), ...(sizes as Record<string, J>) }
}

export const graphsOf = (key: string, bound: BoundModel, store: GraphStore): J =>
  Object.fromEntries(VARIANTS.map((v) => [v.name, graphOf(key, bound, v, store)]))

// ----- a run's posterior -----

/** The diagnostics that are counts and reasons (compared exactly); `change` is a float and goes with the numbers. */
const EXACT_DIAGNOSTICS = [
  'sweeps',
  'updates',
  'improper',
  'improperMessages',
  'repaired',
  'unresolved',
  'restarts',
  'reason',
  'evidenceUndefinedAt',
  'variables',
  'factors',
] as const

/** The verdict and numbers of a posterior: its status, diagnostics, the beliefs by site and the evidence. */
export function posteriorOf(p: Posterior): { readonly exact: Record<string, J>; readonly num: Record<string, J> } {
  const diag = p.diagnostics as Record<string, unknown>
  const exact: Record<string, J> = { status: p.status, sites: [...p.sites] }
  for (const k of EXACT_DIAGNOSTICS) if (diag[k] !== undefined) exact[k] = plain(diag[k])
  const beliefs: Record<string, J> = {}
  for (const site of p.sites) {
    try {
      beliefs[site] = { mean: toFlat(p.mean(site)), sd: toFlat(p.stddev(site)) } as unknown as J
    } catch (e) {
      exact[`beliefError:${site}`] = e instanceof Error ? e.message : String(e)
    }
  }
  const num: Record<string, J> = { beliefs }
  if (p.logEvidence !== undefined) {
    num['evidence'] = p.logEvidence.value
    exact['evidenceMethod'] = p.logEvidence.method ?? null
  }
  if (typeof diag['change'] === 'number') num['change'] = diag['change']
  const known = new Set<string>([...EXACT_DIAGNOSTICS, 'change'])
  const other = Object.keys(diag).filter((k) => !known.has(k))
  if (other.length > 0) exact['otherDiagnostics'] = other.sort()
  return { exact, num }
}

/** Run a method as `atomRun` does (a session of eager steps) and read its posterior; a run that throws is a result of its own. */
function runPosterior(
  method: ReturnType<typeof epSpecMethod> | ReturnType<typeof epGridMethod>,
  steps: number,
  config: Config,
): { readonly exact: Record<string, J>; readonly num: Record<string, J> } {
  try {
    const s = session(method as never as ReturnType<typeof epSpecMethod>, {
      seed: 1,
      steps,
      mode: 'eager',
      config,
      logs: 'off',
    })
    const post = method.posterior?.(s.state as never, s.config as never) as Posterior | undefined
    if (post === undefined)
      return { exact: { noPosterior: true, sessionStatus: s.status ?? null, steps: s.t }, num: {} }
    const out = posteriorOf(post)
    out.exact['sessionStatus'] = s.status ?? null
    out.exact['steps'] = s.t
    return out
  } catch (e) {
    return { exact: { throws: e instanceof Error ? e.message : String(e) }, num: {} }
  }
}

const verdictJson = (v: {
  readonly applies: boolean
  readonly reason: string
  readonly missing: readonly string[]
}): J => ({
  applies: v.applies,
  reason: v.reason,
  missing: [...v.missing],
})

/** Whether EP runs on the model as it is (`ep`) or inside a grid (`ep-grid`), and the verdict of whichever applies (as `atomVerdict`). */
export function routeOf(bound: BoundModel): { readonly route: string; readonly verdict: J } {
  const ep = epSpecMethod(bound).check?.(analysisOf(bound))
  if (ep?.applies === true) return { route: 'ep', verdict: verdictJson(ep) }
  const grid = epGridMethod(bound).check?.(analysisOf(bound))
  if (grid?.applies === true) return { route: 'ep-grid', verdict: verdictJson(grid) }
  return {
    route: 'none',
    verdict: ep === undefined ? { applies: false, reason: 'no verdict', missing: [] } : verdictJson(ep),
  }
}

export const finish = (kind: Item['kind'], exact: J, num: J): Item => ({
  kind,
  exact,
  num,
  bits: sha256(canonical(num)),
})

// ----- the variational run of an atom, as `vmpCoverage` makes it -----

// These are `vmpCoverage`'s own knobs (ep-coverage.ts); `vmpItem` checks that its run agrees with the coverage row, so a drift fails loudly.
const VMP_KNOBS: Config = { tol: 1e-9, maxSweeps: 300, damping: 0.3, improper: 'damp' }
const VMP_STEADY_KNOBS: Config = { tol: 1e-9, maxSweeps: 1000, damping: 0.7, improper: 'damp' }

// ----- the items -----

/** What `epCoverage()` and `vmpCoverage()` say, by name. */
export interface Coverage {
  readonly atoms: ReadonlyMap<string, J>
  readonly benchmarks: ReadonlyMap<string, J>
  readonly vmp: ReadonlyMap<string, VmpRow>
}

type VmpRow = { readonly row: J; readonly status: string; readonly converged?: boolean; readonly sweeps?: number }

const vmpRowOf = (r: ReturnType<typeof vmpCoverage>['atoms'][number]): VmpRow => ({
  row: plain({
    status: r.status,
    reason: r.reason,
    missing: r.missing,
    epRuns: r.epRuns,
    converged: r.converged,
    sweeps: r.sweeps,
  }),
  status: r.status,
  ...(r.converged === undefined ? {} : { converged: r.converged }),
  ...(r.sweeps === undefined ? {} : { sweeps: r.sweeps }),
})

/** The coverage rows of the library (`epCoverage`, `vmpCoverage` without the accuracy pass); empty maps when `on` is false. */
export function coverageOf(on = true): Coverage {
  if (!on) return { atoms: new Map(), benchmarks: new Map(), vmp: new Map() }
  const ep = epCoverage()
  const vmp = vmpCoverage()
  return {
    atoms: new Map(
      ep.atoms.map((r) => [
        r.name,
        plain({
          status: r.status,
          declared: r.declared,
          reason: r.reason,
          missing: r.missing,
          factors: r.factors,
          group: r.group,
        }),
      ]),
    ),
    benchmarks: new Map(
      ep.benchmarks.map((r) => [
        r.name,
        plain({ status: r.status, via: r.via, reason: r.reason, missing: r.missing, group: r.group }),
      ]),
    ),
    vmp: new Map(vmp.atoms.map((r) => [r.name, vmpRowOf(r)])),
  }
}

/** The items of one atom: its EP run (with its check) and, when the variational method runs it, its vmp run. */
export function atomItems(a: EpAtom, cov: Coverage, store: GraphStore): { readonly atom: Item; readonly vmp?: Item } {
  const key = `atom/${a.name}`
  const problem = a.problem(a.defaults)
  const bound = atomBound(problem)
  const graphs = graphsOf(key, bound, store)
  const { route, verdict } = routeOf(bound)
  const v = verifyAtom(a)
  const run = v.run
  const maxSweeps = typeof a.ep?.['maxSweeps'] === 'number' ? a.ep['maxSweeps'] : 200
  const exact: Record<string, J> = {
    graphs,
    route,
    verdict,
    atomVerdict: verdictJson(atomVerdict(problem)),
    status: run.status,
    ...(cov.atoms.has(a.name) ? { coverage: cov.atoms.get(a.name) as J } : {}),
  }
  const num: Record<string, J> = {}
  if (run.status === 'ran') {
    exact['converged'] = run.converged
    exact['sweeps'] = run.sweeps
    num['beliefs'] = plain(run.beliefs)
    if (run.evidence !== undefined) num['evidence'] = run.evidence
    // The engine's own account of the run (sweeps, restarts, improper counts, status), from a second session with the same settings.
    const method = route === 'ep' ? epSpecMethod(bound) : epGridMethod(bound)
    const post = runPosterior(method, route === 'ep' ? maxSweeps + 2 : 1000, { tol: 1e-9, ...(a.ep ?? {}) })
    exact['posterior'] = post.exact
    num['posterior'] = post.num
  } else exact['reason'] = run.reason
  if (v.check !== undefined)
    exact['check'] = {
      pass: v.check.pass,
      rows: v.check.rows.map((r) => ({ name: r.name, pass: r.pass, proper: r.ep.proper })),
    }
  if (v.check !== undefined)
    num['check'] = plain(
      v.check.rows.map((r) => ({
        name: r.name,
        meanZ: r.meanZ,
        sdRatio: r.sdRatio,
        ks: r.ks,
        ep: { mean: r.ep.mean, sd: r.ep.sd },
        refMean: r.refMean,
        refSd: r.refSd,
      })),
    )
  const atom = finish('atom', exact as J, num as J)
  const covered = cov.vmp.size > 0
  const vmpRow =
    cov.vmp.get(a.name) ?? (covered ? undefined : vmpRowOf(vmpCoverage({ names: [a.name] }).atoms[0] as never))
  if (vmpRow === undefined) return { atom }
  return { atom, vmp: vmpItem(a, problem, bound, vmpRow, v.reference, store) }
}

function vmpItem(
  a: EpAtom,
  problem: ReturnType<EpAtom['problem']>,
  bound: BoundModel,
  row: Coverage['vmp'] extends ReadonlyMap<string, infer R> ? R : never,
  reference: ReturnType<typeof atomReference>,
  store: GraphStore,
): Item {
  const exact: Record<string, J> = {
    coverage: row.row,
    graph: graphOf(`vmp/${a.name}`, bound, VARIANTS[2] as (typeof VARIANTS)[number], store),
  }
  const num: Record<string, J> = {}
  if (row.status === 'runs') {
    try {
      const method = vmpSpecMethod(bound)
      let s = session(method, { seed: 1, steps: 302, mode: 'eager', config: VMP_KNOBS, logs: 'off' })
      if (s.status !== 'converged')
        s = session(method, { seed: 1, steps: 1002, mode: 'eager', config: VMP_STEADY_KNOBS, logs: 'off' })
      if ((s.status === 'converged') !== (row.converged ?? false) || s.t !== (row.sweeps ?? -1))
        throw new Error(
          `the digest's variational knobs are out of step with vmpCoverage for ${a.name}: edit VMP_KNOBS in scripts/ep-digest-core.ts`,
        )
      const post = method.posterior?.(s.state, s.config) as Posterior | undefined
      if (post !== undefined) {
        const p = posteriorOf(post)
        exact['posterior'] = p.exact
        num['posterior'] = p.num
        const beliefs = atomBeliefs(problem, post as never)
        num['beliefs'] = plain(beliefs)
        const direct = problem.quantities.filter((q) => q.derive === undefined).map((q) => q.name)
        const check = atomCheck(problem, beliefs, reference, a.tolerance).rows.filter((r) => direct.includes(r.name))
        num['check'] = plain(check.map((r) => ({ name: r.name, meanZ: r.meanZ, sdRatio: r.sdRatio, ks: r.ks })))
      }
    } catch (e) {
      if (e instanceof Error && e.message.startsWith("the digest's variational knobs")) throw e
      exact['throws'] = e instanceof Error ? e.message : String(e)
    }
  }
  return finish('vmp', exact as J, num as J)
}

/** The item of one benchmark model: its graphs, route, verdict and (when EP runs it) the EP run at the default knobs. */
export function modelItem(m: ReturnType<typeof benchmarkModels>[number], cov: Coverage, store: GraphStore): Item {
  const key = `model/${m.name}`
  const exact: Record<string, J> = {
    tags: [...m.tags],
    ...(cov.benchmarks.has(m.name) ? { coverage: cov.benchmarks.get(m.name) as J } : {}),
  }
  const num: Record<string, J> = {}
  if (m.definition === undefined) {
    exact['program'] = true
    return finish('model', exact as J, num as J)
  }
  try {
    const bound = m.bound()
    exact['graphs'] = graphsOf(key, bound, store)
    const { route, verdict } = routeOf(bound)
    exact['route'] = route
    exact['verdict'] = verdict
    if (route !== 'none') {
      const post = runPosterior(
        route === 'ep' ? epSpecMethod(bound) : epGridMethod(bound),
        route === 'ep' ? 202 : 1000,
        { tol: 1e-9 },
      )
      exact['posterior'] = post.exact
      num['posterior'] = post.num
    }
  } catch (e) {
    exact['throws'] = e instanceof Error ? e.message : String(e)
  }
  return finish('model', exact as J, num as J)
}

// ----- the registry -----

/** The hooks an operator may define, as the contract lists them. */
const HOOKS = ['ep', 'vmp', 'bp', 'tilted', 'logEvidence', 'expectedLog', 'check', 'unsupported'] as const

function operatorJson(key: string, op: FactorOperator): J {
  const rec = op as unknown as Record<string, unknown>
  const ports =
    typeof op.ports === 'function'
      ? '[function of the parameters]'
      : op.ports.map((p) => ({
          name: p.name,
          accepts: [...p.accepts],
          constant: p.constant ?? null,
          observed: p.observed ?? null,
        }))
  const flag = (v: unknown): J => (typeof v === 'function' ? '[function]' : v === undefined ? null : (v as J))
  const known = new Set([
    'name',
    'description',
    'ports',
    'support',
    'readsTarget',
    'needsDensity',
    'power',
    'defines',
    'noRule',
    ...HOOKS,
  ])
  return {
    key,
    name: op.name,
    ports: ports as J,
    support: plain(op.support),
    hooks: HOOKS.filter((h) => typeof rec[h] === 'function'),
    readsTarget: flag(op.readsTarget),
    needsDensity: flag(op.needsDensity),
    power: flag(op.power),
    defines: flag(op.defines),
    noRule: plain(op.noRule ?? null),
    otherMembers: Object.keys(rec)
      .filter((k) => !known.has(k))
      .sort(),
  }
}

/** The registry: the operators by key and what each declares, the defaults and knob spaces of the message-passing methods. */
export function registryDigest(): J {
  const ops = defaultOperators()
  const operators = Object.entries(ops).map(([k, op]) => operatorJson(k, op))
  return {
    operatorNames: Object.keys(ops),
    operators,
    defaults: { ep: plain(epDefaults), vmp: plain(vmpDefaults), bp: plain(bpDefaults) },
    knobs: {
      ep: plain(structural(messagePassingKnobs('ep').config)),
      vmp: plain(structural(messagePassingKnobs('vmp').config)),
      bp: plain(structural(messagePassingKnobs('bp').config)),
      epGrid: plain(structural(epGridKnobs)),
      epMultiStart: plain(structural(epMultiStartKnobs)),
    },
    atomKnobs: Object.fromEntries(
      epAtoms.map((a) => [
        a.name,
        { knobs: plain(structural(a.knobs)), defaults: plain(a.defaults), ep: plain(a.ep ?? null) },
      ]),
    ),
  } as unknown as J
}

// ----- the whole digest -----

/** Whether a key is chosen by `--only` (a prefix of the name, or of the whole key). */
export const chosen = (only: readonly string[] | undefined, key: string): boolean =>
  only === undefined ||
  only.some((o) => key === o || key.startsWith(o) || key.slice(key.indexOf('/') + 1).startsWith(o))

/**
 * Compute the digest: every atom, every benchmark model, the variational runs and the registry, serially in this process.
 *
 * @param options - `only` restricts to names (or prefixes); `progress` is told each item as it finishes.
 * @returns The digest and the graphs (canonical JSON by key and variant).
 */
export async function computeDigest(
  options: {
    readonly only?: readonly string[]
    /** Whether to run `epCoverage` and `vmpCoverage` for the coverage rows (about 30 s; default when no `only` is given). */
    readonly coverage?: boolean
    readonly progress?: (key: string, ms: number) => void
    readonly extra?: readonly { readonly key: string; readonly run: (store: GraphStore) => Item | Promise<Item> }[]
  } = {},
): Promise<{ readonly digest: Digest; readonly graphs: GraphStore }> {
  const graphs: GraphStore = new Map()
  const items: Record<string, Item> = {}
  const clock = (): number => Date.now()
  const timed = async (key: string, f: () => void | Promise<void>): Promise<void> => {
    const t = clock()
    await f()
    options.progress?.(key, clock() - t)
  }
  const cov = coverageOf(options.coverage ?? options.only === undefined)
  for (const a of epAtoms) {
    if (!chosen(options.only, `atom/${a.name}`)) continue
    await timed(`atom/${a.name}`, () => {
      const r = atomItems(a as EpAtom, cov, graphs)
      items[`atom/${a.name}`] = r.atom
      if (r.vmp !== undefined) items[`vmp/${a.name}`] = r.vmp
    })
  }
  for (const m of benchmarkModels()) {
    if (!chosen(options.only, `model/${m.name}`)) continue
    await timed(`model/${m.name}`, () => {
      items[`model/${m.name}`] = modelItem(m, cov, graphs)
    })
  }
  for (const e of options.extra ?? []) {
    if (!chosen(options.only, e.key)) continue
    await timed(e.key, async () => {
      items[e.key] = await e.run(graphs)
    })
  }
  const sorted = Object.fromEntries(
    Object.keys(items)
      .sort()
      .map((k) => [k, items[k] as Item]),
  )
  return { digest: { format: 1, items: sorted, registry: registryDigest() }, graphs }
}

/** The digest as the committed file keeps it: numbers rounded to 12 significant digits (the `bits` hash keeps the full precision). */
export function compact(d: Digest): Digest {
  return {
    format: 1,
    registry: d.registry,
    items: Object.fromEntries(
      Object.entries(d.items).map(([k, it]) => [
        k,
        { kind: it.kind, exact: it.exact, num: rounded(it.num), bits: it.bits },
      ]),
    ),
  }
}

/** The text of a digest as written (sorted keys, one item per line, so a diff of the file reads by item). */
export function digestText(d: Digest): string {
  const items = Object.entries(d.items).map(([k, it]) => `${JSON.stringify(k)}:${canonical(it)}`)
  return `{"format":1,\n"registry":${canonical(d.registry)},\n"items":{\n${items.join(',\n')}\n}}\n`
}

// ----- comparing -----

/** The tolerances of the tolerant comparison. */
export interface Tolerances {
  /** Relative, on beliefs (means and sds). */
  readonly belief: number
  /** Absolute floor under the relative tolerance of a belief (a mean at zero). */
  readonly floor: number
  /** Absolute, on a log evidence. */
  readonly evidence: number
  /** Absolute and relative, on the check statistics (z-scores, sd ratios, KS distances). */
  readonly check: number
}

export const defaultTolerances: Tolerances = { belief: 1e-9, floor: 1e-12, evidence: 1e-12, check: 1e-9 }

/** One thing that moved: the item, the field's path, and the old and new values. */
export interface Move {
  readonly item: string
  readonly path: string
  readonly was: string
  readonly now: string
  /** `exact` fields differ in content; `number` ones differ by more than the tolerance (the amount is in `by`); `numbers-rounded` ones differ in the committed digits only. */
  readonly level: 'exact' | 'number' | 'rounded' | 'bits'
  readonly by?: number
}

const short = (v: unknown): string => {
  const s = typeof v === 'string' ? v : canonical(v)
  return s.length > 160 ? `${s.slice(0, 157)}...` : s
}

/** Walk two plain values and list the paths where they differ (arrays by index; arrays of objects with a `name` or `id` by that key). */
function walk(
  a: J | undefined,
  b: J | undefined,
  path: string,
  out: (path: string, a: J | undefined, b: J | undefined) => void,
): void {
  if (
    a === undefined ||
    b === undefined ||
    a === null ||
    b === null ||
    typeof a !== 'object' ||
    typeof b !== 'object'
  ) {
    if (!(a === b || (typeof a === 'number' && typeof b === 'number' && Object.is(a, b)))) out(path, a, b)
    return
  }
  if (Array.isArray(a) !== Array.isArray(b)) {
    out(path, a, b)
    return
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) {
      out(`${path}.length`, a.length, b.length)
      return
    }
    for (let i = 0; i < a.length; i++) walk(a[i], b[i], `${path}[${i}]`, out)
    return
  }
  const oa = a as Record<string, J>
  const ob = b as Record<string, J>
  for (const k of [...new Set([...Object.keys(oa), ...Object.keys(ob)])].sort())
    walk(oa[k], ob[k], path === '' ? k : `${path}.${k}`, out)
}

const isEvidence = (path: string): boolean => path.endsWith('evidence')
const isCheck = (path: string): boolean => path.startsWith('check') || path.includes('.check')
const isChange = (path: string): boolean => path.endsWith('change')

/**
 * Compare a fresh digest to a baseline, by item and field.
 *
 * @param base - The committed baseline (rounded).
 * @param now - The fresh digest, compacted the same way.
 * @param tol - The tolerances.
 * @param only - The names compared (the others are left out of both sides).
 * @returns Every move, and the level reached: `bits` (everything identical, full precision), `exact` (everything but the last digits of numbers identical), `tolerant` (numbers within tolerance) or `none`.
 */
export function compare(
  base: Digest,
  now: Digest,
  tol: Tolerances = defaultTolerances,
  only?: readonly string[],
  coverage = only === undefined,
): { readonly level: 'bits' | 'exact' | 'tolerant' | 'none'; readonly moves: Move[]; readonly compared: number } {
  const moves: Move[] = []
  const keys = [...new Set([...Object.keys(base.items), ...Object.keys(now.items)])]
    .filter((k) => chosen(only, k))
    .sort()
  for (const k of keys) {
    const a = base.items[k]
    const b = now.items[k]
    if (a === undefined || b === undefined) {
      moves.push({
        item: k,
        path: '(item)',
        was: a === undefined ? 'absent' : 'present',
        now: b === undefined ? 'absent' : 'present',
        level: 'exact',
      })
      continue
    }
    walk(a.exact, b.exact, 'exact', (path, x, y) => {
      if (!coverage && /(^|\.)coverage(\.|$)/.test(path)) return
      moves.push({ item: k, path, was: short(x), now: short(y), level: 'exact' })
    })
    walk(a.num, b.num, 'num', (path, x, y) => {
      if (typeof x === 'number' && typeof y === 'number') {
        const diff = Math.abs(x - y)
        if (Number.isNaN(x) && Number.isNaN(y)) return
        if (isChange(path)) {
          moves.push({ item: k, path, was: short(x), now: short(y), level: 'rounded', by: diff })
          return
        }
        const ok = isEvidence(path)
          ? diff <= tol.evidence
          : isCheck(path)
            ? diff <= tol.check * Math.max(1, Math.abs(x), Math.abs(y))
            : diff <= tol.belief * Math.max(Math.abs(x), Math.abs(y)) + tol.floor
        moves.push({ item: k, path, was: short(x), now: short(y), level: ok ? 'rounded' : 'number', by: diff })
      } else moves.push({ item: k, path, was: short(x), now: short(y), level: 'exact' })
    })
    if (a.bits !== b.bits && !moves.some((m) => m.item === k))
      moves.push({ item: k, path: 'bits', was: a.bits.slice(0, 12), now: b.bits.slice(0, 12), level: 'bits' })
  }
  if (only === undefined)
    walk(base.registry, now.registry, 'registry', (path, x, y) =>
      moves.push({ item: 'registry', path, was: short(x), now: short(y), level: 'exact' }),
    )
  const level: 'bits' | 'exact' | 'tolerant' | 'none' = moves.some((m) => m.level === 'exact' || m.level === 'number')
    ? 'none'
    : moves.some((m) => m.level === 'rounded')
      ? 'tolerant'
      : moves.some((m) => m.level === 'bits')
        ? 'exact'
        : 'bits'
  return { level, moves, compared: keys.length }
}

// ----- the difference of two graphs -----

const keyOf = (el: J): string | undefined => {
  if (el === null || typeof el !== 'object' || Array.isArray(el)) return undefined
  const o = el as Record<string, J>
  const k = o['id'] ?? o['name'] ?? o['site']
  return typeof k === 'string' ? k : undefined
}

/**
 * The paths at which two graph JSONs differ, for a report: arrays of objects that carry an `id`, `name` or `site` are matched by it
 * (a factor that moved is one change, not a shift of every later index), the rest by position.
 *
 * @param a - The old graph.
 * @param b - The new graph.
 * @param limit - The most lines returned.
 * @returns Lines such as `factors[factor:y].params.exposure: 1 -> 2`, `variables[w]: removed`.
 */
export function diffPaths(a: J, b: J, limit = 40): string[] {
  const out: string[] = []
  const go = (x: J | undefined, y: J | undefined, path: string): void => {
    if (out.length >= limit) return
    if (x === undefined) return void out.push(`${path}: added ${short(y)}`)
    if (y === undefined) return void out.push(`${path}: removed ${short(x)}`)
    if (x === null || y === null || typeof x !== 'object' || typeof y !== 'object') {
      if (x !== y) out.push(`${path}: ${short(x)} -> ${short(y)}`)
      return
    }
    if (Array.isArray(x) !== Array.isArray(y)) return void out.push(`${path}: ${short(x)} -> ${short(y)}`)
    if (Array.isArray(x) && Array.isArray(y)) {
      const kx = x.map(keyOf)
      const ky = y.map(keyOf)
      const keyed =
        kx.every((k) => k !== undefined) &&
        ky.every((k) => k !== undefined) &&
        new Set(kx).size === x.length &&
        new Set(ky).size === y.length
      if (keyed) {
        const mx = new Map(x.map((e, i) => [kx[i] as string, e]))
        const my = new Map(y.map((e, i) => [ky[i] as string, e]))
        for (const k of new Set([...mx.keys(), ...my.keys()])) go(mx.get(k), my.get(k), `${path}[${k}]`)
        if (kx.join() !== ky.join() && [...mx.keys()].every((k) => my.has(k)) && mx.size === my.size)
          out.push(`${path}: the same members in another order`)
      } else {
        if (x.length !== y.length) out.push(`${path}.length: ${x.length} -> ${y.length}`)
        for (let i = 0; i < Math.min(x.length, y.length); i++) go(x[i], y[i], `${path}[${i}]`)
      }
      return
    }
    const ox = x as Record<string, J>
    const oy = y as Record<string, J>
    for (const k of [...new Set([...Object.keys(ox), ...Object.keys(oy)])].sort())
      go(ox[k], oy[k], path === '' ? k : `${path}.${k}`)
  }
  go(a, b, '')
  return out
}
