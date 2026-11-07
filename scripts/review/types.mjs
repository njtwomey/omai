/**
 * Finds type definitions written more than once: the same name in several files, and different names for the same
 * structure (vectors, matrices, tensor-likes, points), across the packages and apps; then how the numeric parameter
 * types are spelled in the engine's signatures, per domain.
 *
 *   node scripts/review/types.mjs [minimum files for a structure = 2]
 */
import { readFileSync } from 'node:fs'
import { relative } from 'node:path'
import { domainOf, listFor, ROOT, workspaceSource } from './_shared.mjs'

const MIN = Number(process.argv[2] ?? 2)
const strip = (s) => s.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/\/\/[^\n]*/g, ' ')
const norm = (s) =>
  strip(s)
    .replace(/\breadonly\s+/g, '')
    .replace(/ReadonlyArray<([^<>]+)>/g, '$1[]')
    .replace(/Array<([^<>]+)>/g, '$1[]')
    .replace(/\s+/g, ' ')
    .replace(/[;,]\s*}/g, ' }')
    .replace(/\s*([<>{}()[\]|&=:,;])\s*/g, '$1')
    .replace(/[;,]$/, '')
    .trim()
const members = (body) =>
  strip(body)
    .split(/[;\n]/)
    .map((m) => norm(m))
    .filter(Boolean)
    .sort()
    .join(';')

const defs = []
for (const f of workspaceSource()) {
  const text = readFileSync(f, 'utf8')
  const where = relative(ROOT, f)
  for (const m of text.matchAll(
    /^(export )?type ([A-Za-z0-9_]+)(<[^=\n]*>)? =\s*([\s\S]*?)(?=\n(?:export |type |interface |const |function |class |\/\*\*|\/\/|import )|\n\n|$)/gm,
  )) {
    defs.push({ name: m[2], exported: Boolean(m[1]), where, kind: 'type', shape: norm(m[4]), generic: Boolean(m[3]) })
  }
  for (const m of text.matchAll(
    /^(export )?interface ([A-Za-z0-9_]+)(<[^{\n]*>)?(?: extends ([^{\n]+))? \{([\s\S]*?)\n\}/gm,
  )) {
    defs.push({
      name: m[2],
      exported: Boolean(m[1]),
      where,
      kind: 'interface',
      shape: `${m[4] ?? ''}{${members(m[5])}}`,
      generic: Boolean(m[3]),
    })
  }
}
console.log(
  `TYPE DEFINITIONS: ${defs.length} (${defs.filter((d) => d.exported).length} exported) in ${new Set(defs.map((d) => d.where)).size} files`,
)

// ---- the same name in several files ----
const byName = new Map()
for (const d of defs) listFor(byName, d.name).push(d)
const sameName = [...byName].filter(([, v]) => new Set(v.map((d) => d.where)).size > 1)
const identical = sameName.filter(([, v]) => new Set(v.map((d) => d.shape)).size === 1)
console.log(
  `\nSAME NAME, SEVERAL FILES: ${sameName.length} names (${identical.length} with identical structure, ${sameName.length - identical.length} with different structures)`,
)
console.log('  identical structure, largest first (the same type written again):')
for (const [n, v] of identical.sort((a, b) => b[1].length - a[1].length).slice(0, 14)) {
  console.log(
    `    ${n} x${v.length}: ${v[0].shape.slice(0, 60)} | ${v
      .slice(0, 3)
      .map((d) => d.where.replace(/^packages\//, ''))
      .join(', ')}`,
  )
}

// ---- different names, the same structure ----
const byShape = new Map()
for (const d of defs) if (d.shape.length >= 8 && !d.generic) listFor(byShape, d.shape).push(d)
const alias = [...byShape].filter(
  ([, v]) => new Set(v.map((d) => d.where)).size >= MIN && new Set(v.map((d) => d.name)).size > 1,
)
console.log(`\nSAME STRUCTURE UNDER SEVERAL NAMES in ${MIN}+ files: ${alias.length}`)
for (const [shape, v] of alias.sort((a, b) => b[1].length - a[1].length).slice(0, 20)) {
  console.log(`  x${v.length} ${shape.slice(0, 70)} <- ${[...new Set(v.map((d) => d.name))].slice(0, 6).join(', ')}`)
}

// ---- the numeric shapes: vectors, matrices, tensor-likes, points ----
const numeric = defs.filter(
  (d) =>
    /^(readonly )?(\(?number\)?\[\]|number\[\]\[\]|\[number(,number)+\]|Float64Array|ArrayLike<number>|Tensor|TensorLike|Value)/.test(
      d.shape,
    ) || /^(Vec|Vector|Mat|Matrix|Point|Rows?|Tensor|Samples?)\d*$/.test(d.name),
)
const groups = new Map()
for (const d of numeric) listFor(groups, d.shape.slice(0, 60)).push(d)
console.log(
  `\nNUMERIC-SHAPED TYPES (vector, matrix, tuple, tensor-like): ${numeric.length} definitions in ${groups.size} shapes`,
)
for (const [shape, v] of [...groups].sort((a, b) => b[1].length - a[1].length).slice(0, 14)) {
  console.log(`  x${v.length} ${shape}: ${[...new Set(v.map((d) => d.name))].slice(0, 8).join(', ')}`)
}

// ---- how the engine spells numeric parameters, per domain (counts of the spelling in source) ----
const SPELLINGS = {
  'number[]': /\bnumber\[\]/g,
  'number[][]': /\bnumber\[\]\[\]|\(readonly number\[\]\)\[\]|readonly \(readonly number\[\]\)\[\]/g,
  Float64Array: /\bFloat64Array\b/g,
  'ArrayLike<number>': /\bArrayLike<number>/g,
  TensorLike: /\bTensorLike\b/g,
  Value: /:\s*Value\b/g,
}
const perDomain = {}
for (const f of workspaceSource().filter((f) => f.includes('packages/engine/src/lib/'))) {
  const text = readFileSync(f, 'utf8')
  const domain = domainOf(f)
  perDomain[domain] ??= {}
  const row = perDomain[domain]
  for (const [k, re] of Object.entries(SPELLINGS)) row[k] = (row[k] ?? 0) + (text.match(re) ?? []).length
}
const totals = Object.fromEntries(
  Object.keys(SPELLINGS).map((k) => [k, Object.values(perDomain).reduce((a, r) => a + (r[k] ?? 0), 0)]),
)
console.log(
  `\nSPELLINGS in engine source (occurrences): ${Object.entries(totals)
    .map(([k, n]) => `${k} ${n}`)
    .join(', ')}`,
)
console.log('  per core domain (number[] / number[][] / Float64Array / ArrayLike<number> / TensorLike / Value):')
for (const d of [
  'tensor',
  'numerics',
  'optim',
  'dists',
  'stats',
  'metrics',
  'kernels',
  'losses',
  'data',
  'autodiff',
  'random',
]) {
  const r = perDomain[d] ?? {}
  console.log(
    `    ${d.padEnd(9)} ${Object.keys(SPELLINGS)
      .map((k) => String(r[k] ?? 0).padStart(5))
      .join(' ')}`,
  )
}
