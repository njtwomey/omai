/**
 * Scans the core domains for: generic-named helpers defined in several files, constants defined identically in several
 * files, imports that break the layering, and the density of type assertions.
 *
 *   node scripts/review/scan.mjs
 */
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, join, relative } from 'node:path'
import { CORE, CORE_ORDER, domainOf, engineSource, LIB, listFor, PACKAGE, relOf } from './_shared.mjs'

const GENERIC = new Set(
  'dot norm axpy dscal dnrm2 ddot dgemv sum mean clamp cumsum outer transpose matmul argmax argmin linspace range zeros identity eye logsumexp softmax sigmoid median variance maxOf minOf sumOf meanOf dist euclidean sq square lerp sign toArray flatten arange repeat swap copy'.split(
    ' ',
  ),
)
const source = engineSource()
const text = new Map(source.map((f) => [f, readFileSync(f, 'utf8')]))
const core = source.filter((f) => CORE.has(domainOf(f)))
const domains = new Set(readdirSync(LIB).filter((e) => statSync(join(LIB, e)).isDirectory()))

console.log('B. GENERIC-NAMED HELPERS defined in several files (shared modules: blas.ts and tensor/ops):')
const local = new Map()
for (const f of source) {
  for (const m of text
    .get(f)
    .matchAll(/^(?:export )?(?:const|function) ([A-Za-z0-9_]+)\b(?=\s*(?:=\s*(?:\(|<|async)|\())/gm)) {
    if (GENERIC.has(m[1])) listFor(local, m[1]).push(relOf(f))
  }
}
for (const [name, files] of [...local].filter(([, v]) => v.length > 1).sort((a, b) => b[1].length - a[1].length)) {
  console.log(`  ${name} x${files.length}: ${files.slice(0, 6).join(', ')}${files.length > 6 ? ', ...' : ''}`)
}

console.log('\nC. NAMED NUMERIC CONSTANTS defined identically in 3+ files:')
const constants = new Map()
for (const f of source) {
  for (const m of text
    .get(f)
    .matchAll(/^(?:export )?const ([A-Z][A-Z0-9_]*)(?:: number)? = (-?\d[\d_.]*e-?\d+|\d+\.\d+)\b/gm)) {
    listFor(constants, `${m[1]} = ${m[2]}`).push(relOf(f))
  }
}
for (const [k, files] of [...constants].filter(([, v]) => v.length >= 3).sort((a, b) => b[1].length - a[1].length)) {
  console.log(`  ${k} x${files.length}: ${files.slice(0, 4).join(', ')}${files.length > 4 ? ', ...' : ''}`)
}

const edges = []
for (const f of core) {
  for (const m of text.get(f).matchAll(/from '((?:\.\.\/)+)([a-z0-9-]+)\/[^']*'/g)) {
    const to = relative(LIB, join(dirname(f), m[1], m[2])).split('/')[0]
    if (domains.has(to) && to !== domainOf(f)) edges.push({ from: domainOf(f), to, file: relOf(f) })
  }
}
const edgeGroups = (list) => {
  const groups = new Map()
  for (const e of list) listFor(groups, `${e.from} -> ${e.to}`).push(e.file)
  return [...groups].map(([k, v]) => [k, [...new Set(v)]]).sort((a, b) => b[1].length - a[1].length)
}
const down = (e) => CORE_ORDER.indexOf(e.to) > CORE_ORDER.indexOf(e.from)
const outside = edges.filter((e) => !CORE.has(e.to))
const against = edges.filter((e) => CORE.has(e.to) && down(e))
if (PACKAGE === 'engine') {
  console.log(
    `\nD. LAYERING: core files importing a constructed domain: ${outside.length}; against the order: ${against.length}`,
  )
  for (const [k, files] of edgeGroups(outside)) console.log(`  ${k} (${files.length}): ${files.slice(0, 3).join(', ')}`)
  console.log(`  -- against the order (${CORE_ORDER.join(' < ')}):`)
  for (const [k, files] of edgeGroups(against)) console.log(`  ${k} (${files.length}): ${files.slice(0, 3).join(', ')}`)
}

const density = core
  .map((f) => {
    const t = text.get(f)
    const lines = t.split('\n').length
    const casts = (t.match(/\sas (?!const\b)(?!\{)/g) ?? []).length
    return { file: relOf(f), casts, lines, unknown: (t.match(/as unknown as/g) ?? []).length }
  })
  .filter((x) => x.lines > 60)
  .map((x) => ({ ...x, per100: +((100 * x.casts) / x.lines).toFixed(1) }))
  .sort((a, b) => b.per100 - a.per100)
const sum = (key) => density.reduce((a, x) => a + x[key], 0)
console.log(
  `\nE. CAST DENSITY (' as T' per 100 lines, core files over 60 lines): ${sum('casts')} casts, ${sum('unknown')} 'as unknown as'`,
)
for (const x of density.slice(0, 12))
  console.log(`  ${x.per100}/100 lines  ${x.casts} casts in ${x.lines} lines: ${x.file}`)
