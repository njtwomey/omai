/**
 * The rules behind `scripts/layers.ts`, as data and pure functions, so the rules can be tested on fixture edges
 * (`scripts/layers-rules.test.ts`). Nothing here reads the file system.
 *
 * There are three tiers of domains (the folders of `packages/engine/src/lib`), and the rules are the dependency facts
 * of the code, not a total order:
 * - the foundation (`FOUNDATION`): nine domains that each import a short, listed set of the others and nothing
 *   above them. The corrected facts (code review FD-17): `context` sits above `trace` (it builds sessions from the
 *   tape), `scene` sits at the bottom beside `base` (it is plain data), and `autodiff` takes `scene` for its graph
 *   description;
 * - the machinery (`MACHINERY`): numerics, distributions, optimisation, losses, kernels, tests and metrics, the
 *   information measures and the data layer. They may import the foundation and one another (without a cycle: that
 *   is the other check) and nothing else, so an estimator, a signal or a graph can build on them and never the
 *   reverse. The few imports that go the other way are listed in `EXCEPTIONS`, each with its reason;
 * - everything else (learn, signal, graph, text, nn, rl ...) is constructed from the two tiers and may import any
 *   domain that does not lead back to it.
 *
 * A file-level import cycle inside one domain is reported too (value imports only: `import type` is erased, so it
 * cannot misorder module loading). The cycles that exist today are written out in `FILE_CYCLE_ALLOWLIST`.
 */

/** One import between two files of the engine, with `from` and `to` as paths relative to `packages/engine/src/lib`. */
export interface ImportEdge {
  readonly from: string
  readonly to: string
  /** Whether the import is `import type` / `export type` (erased at run time). */
  readonly typeOnly: boolean
  /** A human-readable example of the import, for messages. */
  readonly where: string
}

/** What each foundation domain may import: the dependency facts, checked in both directions of the tier. */
export const FOUNDATION: ReadonlyMap<string, readonly string[]> = new Map([
  ['base', []],
  ['scene', ['base']],
  ['exposition', ['base', 'scene']],
  ['tensor', ['base']],
  ['random', ['base', 'tensor']],
  ['space', ['base', 'random', 'tensor']],
  ['trace', ['base', 'random', 'space', 'tensor']],
  ['autodiff', ['base', 'random', 'scene', 'tensor']],
  ['context', ['base', 'random', 'tensor', 'trace']],
])

/**
 * The machinery tier: may import the foundation and one another. `information` belongs here (entropy and mutual
 * information are what `stats` and `losses` build on; it imports only the foundation and `numerics`).
 */
export const MACHINERY: ReadonlySet<string> = new Set([
  'numerics',
  'dists',
  'information',
  'optim',
  'losses',
  'kernels',
  'stats',
  'metrics',
  'data',
])

/** An import that goes against the tiers on purpose: a file (or folder) of the machinery, the domain it reaches. */
export interface Exception {
  /** A file or folder, relative to `packages/engine/src/lib`. */
  readonly from: string
  readonly to: string
  readonly reason: string
}

/** The machinery imports of a constructed domain that exist because moving the code down would cost more than it buys. */
export const EXCEPTIONS: readonly Exception[] = [
  {
    from: 'metrics/audio.ts',
    to: 'signal',
    reason: 'audio quality metrics (STOI, PESQ-style, MCD) compute on `stft`, `firwin`, `resamplePoly` and `dtw`',
  },
  {
    from: 'metrics/image-quality.ts',
    to: 'signal',
    reason: 'SSIM and relatives filter planes with `signal`/`image` (`uniformFilter`, `gaussianFilter`, `correlate1d`)',
  },
  {
    from: 'metrics/text.ts',
    to: 'text',
    reason: 'BLEU, ROUGE, TER and relatives count n-grams and edits with the helpers of `text`',
  },
  {
    from: 'kernels/structured.ts',
    to: 'graph',
    reason:
      "graph kernels (heat, Weisfeiler-Lehman, Laplacian) are defined on `graph`'s `Graph` and its spectral operators",
  },
]

/** A strongly connected set of files inside one domain that is known and accepted for now. */
export interface AllowedCycle {
  /** The files of the cycle (relative to `packages/engine/src/lib`), in any order. */
  readonly files: readonly string[]
  readonly reason: string
}

const OPS = ['_coerce', '_factories', 'arithmetic', 'compare', 'indexing', 'math', 'reduce', 'shape'] as const

/**
 * The file-level cycles that exist today, written out. A cycle not listed here fails the check; a listed one that no
 * longer exists as a whole fails too, so the list only shrinks.
 */
export const FILE_CYCLE_ALLOWLIST: readonly AllowedCycle[] = [
  {
    files: [
      'tensor/cpu.ts',
      'tensor/_cpu/_util.ts',
      'tensor/_cpu/elementwise.ts',
      'tensor/_cpu/reduce.ts',
      'tensor/_cpu/sort.ts',
      'tensor/_cpu/structure.ts',
    ],
    reason:
      '`_cpu/_util.wrap` stamps the `cpu` object on every tensor it builds and `cpu` is the object made of those ' +
      'kernels; `wrap` reads `cpu` only when called, never while the modules load',
  },
  {
    files: ['tensor/batching.ts', ...OPS.map((o) => `tensor/ops/${o}.ts`)],
    reason:
      'the operations are written in terms of one another (`arithmetic` uses `compare` and `math`, which use ' +
      '`arithmetic`) and `batching.ts` uses the shape operations its batching rules need; the module-level ' +
      '`unaryPrimitive(...)` calls load only because every cross-module use sits inside a function. Breaking it means a ' +
      'leaf module of the shared primitives (P3.15, not done)',
  },
  {
    files: ['search/planning/graphplan.ts', 'search/planning/strips.ts'],
    reason:
      'found when the file-level check was added: `strips.ts` imports `mutexPairs` from `graphplan.ts`, which imports `validatePlan` from `strips.ts`; outside P3.15 (`mutexPairs` could move to `_relaxed.ts`)',
  },
  {
    files: ['symbolic/_canon.ts', 'symbolic/_functions.ts', 'symbolic/_numeric.ts', 'symbolic/_order.ts'],
    reason:
      'found when the file-level check was added: the canonical constructors (`_canon`), the function table (`_functions`), the numeric evaluator and the term order call one another; outside P3.15, not examined',
  },
  {
    files: ['symbolic/matrices.ts', 'symbolic/polynomials.ts'],
    reason:
      'found when the file-level check was added: `matrices.ts` imports `exactRoots` from `polynomials.ts`, which imports `det` from `matrices.ts`; outside P3.15',
  },
]

const domainOf = (file: string): string => file.split('/')[0] ?? ''

/** Whether the exception covers an import of `to` from `from`: the same file, or a file inside the exception's folder. */
function covers(e: Exception, from: string, to: string): boolean {
  return e.to === to && (from === e.from || from.startsWith(`${e.from}/`))
}

/** The tier rule applied to every import: the violations, and the exceptions that excused something. */
function applyTiers(edges: readonly ImportEdge[]): { errors: string[]; used: Set<Exception> } {
  const errors: string[] = []
  const used = new Set<Exception>()
  for (const edge of edges) {
    const from = domainOf(edge.from)
    const to = domainOf(edge.to)
    if (from === to || to === '') continue
    const allowed = FOUNDATION.get(from)
    let message: string | null = null
    if (allowed !== undefined) {
      if (!allowed.includes(to)) message = `foundation domain '${from}' may import only [${allowed.join(', ')}]`
    } else if (MACHINERY.has(from) && !FOUNDATION.has(to) && !MACHINERY.has(to)) {
      message = `machinery domain '${from}' may import only the foundation and the machinery`
    }
    if (message === null) continue
    const exception = EXCEPTIONS.find((e) => covers(e, edge.from, to))
    if (exception !== undefined) used.add(exception)
    else errors.push(`${edge.where}: ${message}, not '${to}' (exceptions are listed in scripts/layers-rules.ts)`)
  }
  return { errors, used }
}

/**
 * The tier violations among imports between different domains.
 *
 * @param edges - Every import between files of the engine (edges inside one domain are ignored).
 * @returns One message per violation that no exception covers; empty when the tiers hold.
 */
export function tierViolations(edges: readonly ImportEdge[]): string[] {
  return applyTiers(edges).errors
}

/**
 * The exceptions that no import uses any more: the table must stay true, so each is a failure.
 *
 * @param edges - Every import between files of the engine.
 * @returns One message per unused exception.
 */
export function staleExceptions(edges: readonly ImportEdge[]): string[] {
  const { used } = applyTiers(edges)
  return EXCEPTIONS.filter((e) => !used.has(e)).map(
    (e) => `stale layering exception ${e.from} -> ${e.to} (${e.reason}): remove it from EXCEPTIONS`,
  )
}

/** A directed graph of named nodes, with one example import per edge. */
export type Graph = Map<string, Map<string, string>>

/** Add the edge `from -> to` (the first example import is kept). */
export function addEdge(graph: Graph, from: string, to: string, where: string): void {
  const out = graph.get(from) ?? new Map<string, string>()
  if (!out.has(to)) out.set(to, where)
  graph.set(from, out)
}

/** A strongly connected component of more than one node: its members and the shortest path that returns to its start. */
export interface Cycle {
  readonly members: readonly string[]
  readonly path: readonly string[]
}

/** Every cycle of a graph, one per strongly connected component (Tarjan). */
export function cycles(graph: Graph): Cycle[] {
  const index = new Map<string, number>()
  const low = new Map<string, number>()
  const stack: string[] = []
  const onStack = new Set<string>()
  const found: Cycle[] = []
  let next = 0
  const connect = (v: string): void => {
    index.set(v, next)
    low.set(v, next)
    next++
    stack.push(v)
    onStack.add(v)
    for (const w of (graph.get(v) ?? new Map()).keys()) {
      if (!index.has(w)) {
        connect(w)
        low.set(v, Math.min(low.get(v) as number, low.get(w) as number))
      } else if (onStack.has(w)) low.set(v, Math.min(low.get(v) as number, index.get(w) as number))
    }
    if (low.get(v) !== index.get(v)) return
    const component = new Set<string>()
    let w: string
    do {
      w = stack.pop() as string
      onStack.delete(w)
      component.add(w)
    } while (w !== v)
    if (component.size > 1) found.push({ members: [...component].sort(), path: shortestCycle(graph, v, component) })
  }
  for (const v of [...graph.keys()].sort()) if (!index.has(v)) connect(v)
  return found
}

/** The shortest path from `start` back to itself inside a strongly connected component (breadth first). */
function shortestCycle(graph: Graph, start: string, component: ReadonlySet<string>): string[] {
  const previous = new Map<string, string>()
  const queue = [start]
  while (queue.length > 0) {
    const v = queue.shift() as string
    for (const w of [...(graph.get(v) ?? new Map()).keys()].sort()) {
      if (!component.has(w)) continue
      if (w === start) {
        const path: string[] = []
        for (let u = v; u !== start; u = previous.get(u) as string) path.unshift(u)
        return [start, ...path, start]
      }
      if (!previous.has(w)) {
        previous.set(w, v)
        queue.push(w)
      }
    }
  }
  return [start, start]
}

/** The graph of value imports between files of one domain (edges across domains and type-only imports are left out). */
export function fileGraph(edges: readonly ImportEdge[]): Graph {
  const graph: Graph = new Map()
  for (const e of edges) {
    if (e.typeOnly || e.from === e.to || domainOf(e.from) !== domainOf(e.to)) continue
    addEdge(graph, e.from, e.to, e.where)
  }
  return graph
}

/**
 * The file-level cycles inside domains that are not in the allowlist, and the allowlisted cycles that are gone.
 *
 * @param edges - Every import between files of the engine.
 * @param allowed - The accepted cycles (defaults to `FILE_CYCLE_ALLOWLIST`).
 * @returns The cycles found, each flagged `allowed` or not, and a message for every allowlist entry that no longer
 *   exists as a whole.
 */
export function fileCycleReport(
  edges: readonly ImportEdge[],
  allowed: readonly AllowedCycle[] = FILE_CYCLE_ALLOWLIST,
): { readonly found: readonly (Cycle & { readonly allowed: AllowedCycle | null })[]; readonly stale: string[] } {
  const graph = fileGraph(edges)
  const key = (files: readonly string[]): string => [...files].sort().join('\n')
  const table = new Map(allowed.map((a) => [key(a.files), a] as const))
  const found = cycles(graph).map((c) => ({ ...c, allowed: table.get(key(c.members)) ?? null }))
  const present = new Set(found.map((c) => key(c.members)))
  const stale = allowed
    .filter((a) => !present.has(key(a.files)))
    .map(
      (a) =>
        `allowlisted file cycle is gone or changed: ${[...a.files].sort().join(', ')}; remove it from FILE_CYCLE_ALLOWLIST`,
    )
  return { found, stale }
}
