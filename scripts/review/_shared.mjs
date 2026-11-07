/**
 * What the review scripts share: where the engine is, which domains are the core and in what order they stack, how to
 * walk and filter its files, and how to turn a function body into comparable tokens and shingles.
 */
import { createHash } from 'node:crypto'
import { readdirSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'

export const ROOT = join(import.meta.dirname, '..', '..')
export const LIB = join(ROOT, 'packages/engine/src/lib')

/** The core, in the order they stack (a first guess at the layering: lower may not import higher). */
export const CORE_ORDER = [
  'base',
  'context',
  'tensor',
  'random',
  'space',
  'trace',
  'autodiff',
  'numerics',
  'optim',
  'scene',
  'exposition',
  'losses',
  'dists',
  'kernels',
  'stats',
  'metrics',
  'data',
]
export const CORE = new Set(CORE_ORDER)

/** Every .ts and .tsx file under a folder, skipping build output and dependencies. */
export function walk(dir, found = []) {
  for (const entry of readdirSync(dir)) {
    if (['node_modules', 'dist', '.generated', 'index', 'fixtures'].includes(entry)) continue
    const path = join(dir, entry)
    if (statSync(path).isDirectory()) walk(path, found)
    else if (/\.tsx?$/.test(entry)) found.push(path)
  }
  return found
}

export const isTest = (f) =>
  /\.(test|property\.test|golden\.test|test-d)\.tsx?$/.test(f) || /examples\.test/.test(f) || /\/testing\//.test(f)
export const isData = (f) => /\.data\.ts$/.test(f) || /\/_data\//.test(f)
export const isBarrel = (f) => /\/index\.ts$/.test(f)

/** The engine's source files: no tests, no generated data, no barrels. */
export const engineSource = () => walk(LIB).filter((f) => !isTest(f) && !isData(f) && !isBarrel(f))
export const domainOf = (f) => relative(LIB, f).split('/')[0]
export const relOf = (f) => relative(LIB, f)

const KEYWORDS = new Set(
  'const let return if else for of in while new function throw true false null undefined number string boolean void readonly as typeof export type interface extends'.split(
    ' ',
  ),
)

/** The tokens of a body with comments dropped, literals folded and identifiers replaced by `I`. */
export function tokens(body) {
  const folded = body
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/\/\/[^\n]*/g, ' ')
    .replace(/'[^']*'|`[^`]*`/g, 'S')
    .replace(/\b\d[\d_.e+-]*\b/g, 'N')
  return (folded.match(/[A-Za-z_]\w*|[^\sA-Za-z_0-9]/g) ?? []).map((t) =>
    /^[A-Za-z_]/.test(t) && !KEYWORDS.has(t) ? 'I' : t,
  )
}

/** The set of hashes of every run of five tokens. */
export function shingles(toks) {
  const found = new Set()
  for (let k = 0; k + 5 <= toks.length; k++) {
    found.add(
      createHash('md5')
        .update(toks.slice(k, k + 5).join(' '))
        .digest('hex')
        .slice(0, 8),
    )
  }
  return found
}

/** The share of the smaller set that the other holds. */
export function overlap(a, b) {
  const [small, large] = a.size < b.size ? [a, b] : [b, a]
  let shared = 0
  for (const h of small) if (large.has(h)) shared++
  return shared / Math.max(1, small.size)
}

/** `a ?? (map.set(key, init), …)` without the expression tricks: the list kept under a key, made on first use. */
export function listFor(map, key) {
  let list = map.get(key)
  if (list === undefined) {
    list = []
    map.set(key, list)
  }
  return list
}
