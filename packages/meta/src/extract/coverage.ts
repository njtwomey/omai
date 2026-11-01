/**
 * Documentation coverage: what the conventions of docs/structure.md section 2 ask for and the source does not yet
 * have. For exports: a summary, every parameter described, `@returns` for a function that returns something, and an
 * `@example` with at least one `// =>` expectation for every function, class and constant of the core. For the tree:
 * a summary for every folder (the doc comment that opens its `index.ts`) and an opening paragraph for every file.
 */
import type { Catalogue, ModuleLayer, SymbolDoc } from '../schema/catalogue.ts'

/** What a symbol can be missing. */
export type CoverageProblem = 'summary' | 'param' | 'returns' | 'example' | 'expectation'

/** One gap in the documentation. */
export interface CoverageIssue {
  readonly symbol: string
  readonly problem: CoverageProblem
  /** The parameter, for `param`. */
  readonly detail: string
  readonly file: string
  readonly line: number
}

/** Coverage over a set of symbols. */
export interface CoverageReport {
  readonly symbols: number
  /** Symbols with no issue at all. */
  readonly complete: number
  readonly issues: readonly CoverageIssue[]
  /** Issue counts by problem. */
  readonly counts: Readonly<Record<CoverageProblem, number>>
}

function returnsSomething(s: SymbolDoc): boolean {
  if (s.kind !== 'function') return false
  const last = s.signature.split('\n').at(-1) ?? ''
  return !/\)\s*:\s*(void|Promise<void>|never|asserts\s[^)]*)\s*$/.test(last)
}

/**
 * Check every symbol of the given layers against the documentation conventions.
 *
 * @param catalogue - The catalogue.
 * @param layers - Which module layers to check (default the core).
 * @returns The issues, and counts.
 */
export function coverageReport(catalogue: Catalogue, layers: readonly ModuleLayer[] = ['core']): CoverageReport {
  const layerOf = new Map(catalogue.modules.map((m) => [m.id, m.layer]))
  const issues: CoverageIssue[] = []
  let symbols = 0
  let complete = 0
  for (const s of catalogue.symbols) {
    if (!layers.includes(layerOf.get(s.module) ?? 'part')) continue
    symbols++
    const before = issues.length
    const add = (problem: CoverageProblem, detail = ''): void => {
      issues.push({ symbol: s.id, problem, detail, file: s.source.file, line: s.source.line })
    }
    if (s.summary === '') add('summary')
    for (const p of s.params) if (!p.name.includes('.') && p.description === '') add('param', p.name)
    if (returnsSomething(s) && s.returns === '') add('returns')
    if (s.kind === 'function' || s.kind === 'class' || s.kind === 'const') {
      const examples = [...s.examples, ...s.members.flatMap((m) => m.examples)]
      if (examples.length === 0) add('example')
      else if (!examples.some((e) => /\/\/\s*=>/.test(e.code))) add('expectation')
    }
    if (issues.length === before) complete++
  }
  const counts = { summary: 0, param: 0, returns: 0, example: 0, expectation: 0 }
  for (const i of issues) counts[i.problem]++
  return { symbols, complete, issues, counts }
}

/** Folders and files without a summary. */
export interface SummaryReport {
  readonly folders: number
  readonly files: number
  /** Repository-relative paths of folders whose `index.ts` opens with no summary. */
  readonly missingFolders: readonly string[]
  /** Repository-relative paths of files that open with no paragraph of their own. */
  readonly missingFiles: readonly string[]
}

/**
 * Check that every folder and file of the given layers says what it is: a folder by the doc comment that opens its
 * `index.ts`, a file by its opening paragraph.
 *
 * @param catalogue - The catalogue.
 * @param layers - Which module layers to check (default all).
 * @returns The counts and the paths without a summary.
 */
export function summaryCoverage(
  catalogue: Catalogue,
  layers: readonly ModuleLayer[] = ['core', 'part'],
): SummaryReport {
  const layerOf = new Map(catalogue.modules.map((m) => [m.id, m.layer]))
  const kept = (module: string): boolean => layers.includes(layerOf.get(module) ?? 'part')
  const folders = catalogue.folders.filter((f) => kept(f.module))
  const files = catalogue.files.filter((f) => kept(f.module))
  return {
    folders: folders.length,
    files: files.length,
    missingFolders: folders.filter((f) => f.summary === '').map((f) => f.path),
    missingFiles: files.filter((f) => f.summary === '').map((f) => f.path),
  }
}
