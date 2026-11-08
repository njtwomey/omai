/**
 * Paths and small helpers shared by the aifn-usage scripts: where ml-concepts, the aifn-engine and the output folder
 * are, how a JSON file is written (sorted, one record per line so diffs stay readable), and the shapes the scripts pass
 * to one another through `docs/aifn/data/`. Reads only; the one place anything is written is `docs/aifn/`.
 */
import { execFileSync } from 'node:child_process'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join, resolve } from 'node:path'

export const ROOT = resolve(import.meta.dirname, '..', '..')
export const DATA = join(ROOT, 'docs', 'aifn', 'data')
export const DOCS = join(ROOT, 'docs', 'aifn')
export const ML = process.env['ML_CONCEPTS'] ?? join(homedir(), 'workspace', 'ml-concepts')
export const AIFN = process.env['AIFN_ENGINE'] ?? join(homedir(), 'workspace', 'aifn-engine')
export const PACKAGES = ['aifn-compute', 'aifn-render', 'aifn-methods'] as const
export type AifnPackage = (typeof PACKAGES)[number]

/** One use of an aifn name in a file. */
export interface ImportUse {
  readonly package: string
  /** The path below the package, `''` for the package root. */
  readonly subpath: string
  /** The imported name; `*` for a namespace or dynamic import. */
  readonly name: string
  readonly typeOnly: boolean
}

/** A symbol used by the notes, as `usage.json` records it. */
export interface UsedSymbol {
  readonly id: string
  readonly package: string
  readonly subpath: string
  readonly name: string
  /** Notes (folders with an `index.mdx`) that reach the symbol directly or through shared files. */
  readonly notes: number
  readonly noteIndexes: readonly number[]
  readonly files: number
  readonly labFiles: number
  readonly siteFiles: number
  readonly scriptFiles: number
  readonly typeOnly: boolean
  readonly examples: readonly string[]
}

/** The shape of `usage.json`. */
export interface UsageData {
  readonly source: { readonly commit: string; readonly aifnVersion: string }
  readonly totals: Record<string, number>
  readonly notes: readonly string[]
  readonly symbols: readonly UsedSymbol[]
  readonly subpaths: readonly { readonly specifier: string; readonly notes: number; readonly symbols: number }[]
  readonly workerCalls: readonly {
    readonly call: string
    readonly notes: number
    readonly noteIndexes: readonly number[]
    readonly examples: readonly string[]
  }[]
  readonly dynamicImports: readonly { readonly specifier: string; readonly files: readonly string[] }[]
  /** Code the notes wrote themselves: `.ts` files below `content/notes`, with the notes that reach them. */
  readonly localFiles: readonly {
    readonly file: string
    readonly lines: number
    readonly notes: number
    readonly aifnImports: number
    readonly exports: readonly string[]
  }[]
}

/** An exported AIFN name at its canonical home. */
export interface AifnSymbol {
  readonly id: string
  readonly name: string
  readonly kind: string
  /** The declaring source file below the aifn-engine root, with the 1-based line. */
  readonly source: string
  readonly summary: string
  /** The entry points other than the home that export it. */
  readonly alsoIn?: readonly string[]
}

export interface AifnSurface {
  readonly commit: string
  readonly entries: readonly { readonly specifier: string; readonly file: string; readonly count: number }[]
  readonly symbols: readonly AifnSymbol[]
}

export function git(dir: string, ...args: string[]): string {
  try {
    return execFileSync('git', ['-C', dir, ...args], { encoding: 'utf8' }).trim()
  } catch {
    return 'unknown'
  }
}

export function readJson<T>(file: string): T {
  return JSON.parse(readFileSync(join(DATA, file), 'utf8')) as T
}

/**
 * Write a JSON file under `docs/aifn/data/`: an object's top-level arrays are written one element per line, the rest
 * indented, so the files stay diffable and small.
 */
export function writeJson(file: string, value: Record<string, unknown>): void {
  const path = join(DATA, file)
  mkdirSync(dirname(path), { recursive: true })
  const parts = Object.entries(value).map(([k, v]) => {
    if (Array.isArray(v) && v.length > 0 && typeof v[0] === 'object') {
      return `  ${JSON.stringify(k)}: [\n${v.map((x) => `    ${JSON.stringify(x)}`).join(',\n')}\n  ]`
    }
    return `  ${JSON.stringify(k)}: ${JSON.stringify(v)}`
  })
  writeFileSync(path, `{\n${parts.join(',\n')}\n}\n`)
}

export function writeText(file: string, text: string): void {
  const path = join(DOCS, file)
  mkdirSync(dirname(path), { recursive: true })
  writeFileSync(path, text)
}

/** A markdown table, its columns padded to the widest cell as the repo's formatter writes them. */
export function table(head: readonly string[], rows: readonly (readonly (string | number)[])[]): string {
  const esc = (s: string | number): string => String(s).replace(/\|/g, '\\|')
  const cells = [head.map(esc), ...rows.map((r) => r.map(esc))]
  const width = head.map((_, i) => Math.max(3, ...cells.map((r) => [...(r[i] ?? '')].length)))
  const line = (r: readonly string[]): string => `| ${r.map((c, i) => c.padEnd(width[i] ?? 0)).join(' | ')} |`
  return [line(cells[0] ?? []), `| ${width.map((w) => '-'.repeat(w)).join(' | ')} |`, ...cells.slice(1).map(line)].join(
    '\n',
  )
}

export const byCountThenName = (a: { n: number; k: string }, b: { n: number; k: string }): number =>
  b.n - a.n || (a.k < b.k ? -1 : a.k > b.k ? 1 : 0)

const STOP = new Set([
  'steps',
  'step',
  'run',
  'state',
  'options',
  'option',
  'model',
  'problem',
  'result',
  'params',
  'checkpoint',
  'snapshot',
  'of',
  'the',
  'and',
  'at',
  'for',
  'name',
  'spec',
  'kind',
])

/** The lower-case words of a camelCase name, without the words every method carries (`steps`, `run`, `options`). */
export function nameTokens(name: string): string[] {
  return name
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t !== '' && !STOP.has(t))
}

/** The names of `pool` closest to `name` by shared words (Jaccard), best first, at or above `threshold`. */
export function nearestNames<T extends { readonly name: string }>(
  name: string,
  pool: readonly T[],
  threshold: number,
  limit = 3,
): { item: T; score: number }[] {
  const mine = nameTokens(name)
  if (mine.length === 0) return []
  return pool
    .map((item) => {
      const theirs = new Set(nameTokens(item.name))
      const inter = mine.filter((x) => theirs.has(x)).length
      return { item, score: inter / (mine.length + theirs.size - inter) }
    })
    .filter((x) => x.score >= threshold)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
}
