import { createHash } from 'node:crypto'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'

const R = join(import.meta.dirname, '..', '..')
const lib = join(R, 'packages/engine/src/lib')
const CORE = [
  'base',
  'tensor',
  'random',
  'space',
  'trace',
  'autodiff',
  'context',
  'numerics',
  'optim',
  'losses',
  'dists',
  'stats',
  'metrics',
  'kernels',
  'data',
  'exposition',
  'scene',
]
const ORDER = [
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
const walk = (d, out = []) => {
  for (const e of readdirSync(d)) {
    if (e === 'node_modules' || e === 'dist' || e === '.generated') continue
    const p = join(d, e)
    statSync(p).isDirectory() ? walk(p, out) : /\.tsx?$/.test(e) && out.push(p)
  }
  return out
}
const isTest = (f) =>
  /\.(test|property\.test|golden\.test|test-d)\.tsx?$/.test(f) || /examples\.test/.test(f) || /\/testing\//.test(f)
const isData = (f) => /\.data\.ts$/.test(f) || /\/_data\//.test(f)
const dom = (f) => relative(lib, f).split('/')[0]
const all = walk(lib).filter((f) => !isData(f))
const src = all.filter((f) => !isTest(f))
const core = src.filter((f) => CORE.includes(dom(f)))
const text = new Map(src.map((f) => [f, readFileSync(f, 'utf8')]))
const rel = (f) => relative(lib, f)
const report = []
const out = (s = '') => {
  report.push(s)
  console.log(s)
}

// ---- A. near-duplicate functions in the core (token shingles) ----
const funcs = []
for (const f of core) {
  const lines = text.get(f).split('\n')
  for (let i = 0; i < lines.length; i++) {
    const m = /^(export )?(async )?function\*? ([A-Za-z0-9_]+)/.exec(lines[i])
    if (!m) continue
    let j = i
    while (j < lines.length && lines[j] !== '}') j++
    if (j - i + 1 < 10) continue
    const toks =
      lines
        .slice(i + 1, j)
        .join(' ')
        .replace(/\/\/[^\n]*/g, ' ')
        .replace(/'[^']*'|`[^`]*`/g, 'S')
        .replace(/\b\d[\d_.e+-]*\b/g, 'N')
        .match(/[A-Za-z_]\w*|[^\sA-Za-z_0-9]/g) ?? []
    const norm = toks.map((t) =>
      /^[A-Za-z_]/.test(t) &&
      !/^(const|let|return|if|else|for|of|in|while|new|function|throw|true|false|null|undefined|number|string|boolean|void|readonly|as|typeof)$/.test(
        t,
      )
        ? 'I'
        : t,
    )
    const sh = new Set()
    for (let k = 0; k + 6 <= norm.length; k++)
      sh.add(
        createHash('md5')
          .update(norm.slice(k, k + 6).join(' '))
          .digest('hex')
          .slice(0, 8),
      )
    funcs.push({ file: f, name: m[3], line: i + 1, len: j - i + 1, sh })
  }
}
const inv = new Map()
for (const [id, fn] of funcs.entries()) for (const h of fn.sh) (inv.get(h) ?? inv.set(h, []).get(h)).push(id)
const pairs = []
funcs.forEach((a, ia) => {
  const cnt = new Map()
  a.sh.forEach((h) => {
    const l = inv.get(h)
    if (l.length < 40)
      l.forEach((ib) => {
        if (ib > ia) cnt.set(ib, (cnt.get(ib) ?? 0) + 1)
      })
  })
  for (const [ib, c] of cnt) {
    const b = funcs[ib]
    const sim = c / Math.min(a.sh.size, b.sh.size)
    if (sim >= 0.8 && a.file !== b.file) pairs.push({ sim: +sim.toFixed(2), a, b })
  }
})
pairs.sort((x, y) => y.sim * Math.min(y.a.len, y.b.len) - x.sim * Math.min(x.a.len, x.b.len))
out(
  `A. NEAR-DUPLICATE FUNCTIONS in the core (>=10 lines, token similarity >= 0.8, different files): ${pairs.length} pairs among ${funcs.length} functions`,
)
for (const p of pairs.slice(0, 25))
  out(
    `  ${p.sim} ${rel(p.a.file)}:${p.a.line} ${p.a.name}(${p.a.len}) ~ ${rel(p.b.file)}:${p.b.line} ${p.b.name}(${p.b.len})`,
  )

// ---- B. local copies of generic helpers ----
const GENERIC = [
  'dot',
  'norm',
  'axpy',
  'dscal',
  'dnrm2',
  'ddot',
  'dgemv',
  'sum',
  'mean',
  'clamp',
  'cumsum',
  'outer',
  'transpose',
  'matmul',
  'argmax',
  'argmin',
  'linspace',
  'range',
  'zeros',
  'identity',
  'eye',
  'logsumexp',
  'softmax',
  'sigmoid',
  'median',
  'variance',
  'maxOf',
  'minOf',
  'sumOf',
  'meanOf',
  'dist',
  'euclidean',
  'sq',
  'square',
  'lerp',
  'sign',
  'isFiniteNumber',
  'assertFinite',
  'toArray',
  'flatten',
  'arange',
  'repeat',
  'swap',
  'copy',
]
const local = {}
for (const f of src) {
  for (const m of text
    .get(f)
    .matchAll(/^(?:export )?(?:const|function) ([A-Za-z0-9_]+)\b(?=\s*(?:=\s*(?:\(|<|async)|\())/gm)) {
    if (GENERIC.includes(m[1])) {
      local[m[1]] ??= []
      local[m[1]].push(rel(f))
    }
  }
}
out(`\nB. GENERIC-NAMED HELPERS defined in several files (name: files; shared modules are blas.ts and tensor/ops):`)
for (const [k, v] of Object.entries(local)
  .filter(([, v]) => v.length > 1)
  .sort((a, b) => b[1].length - a[1].length)
  .slice(0, 20))
  out(`  ${k} x${v.length}: ${v.slice(0, 6).join(', ')}${v.length > 6 ? ', ...' : ''}`)

// ---- C. repeated tolerance/epsilon constants ----
const consts = new Map()
for (const f of src)
  for (const m of text
    .get(f)
    .matchAll(/^(?:export )?const ([A-Z][A-Z0-9_]*)(?:: number)? = (-?\d[\d_.]*e-?\d+|\d+\.\d+)\b/gm)) {
    const k = `${m[1]} = ${m[2]}`
    ;(consts.get(k) ?? consts.set(k, []).get(k)).push(rel(f))
  }
const crep = [...consts.entries()].filter(([, v]) => v.length >= 3).sort((a, b) => b[1].length - a[1].length)
out(`\nC. NAMED NUMERIC CONSTANTS defined identically in 3+ files: ${crep.length}`)
for (const [k, v] of crep.slice(0, 15))
  out(`  ${k} x${v.length}: ${v.slice(0, 4).join(', ')}${v.length > 4 ? ', ...' : ''}`)

// ---- D. layering: core importing constructed, and core edges against the rank order ----
const edges = []
for (const f of core)
  for (const m of text.get(f).matchAll(/from '((?:\.\.\/)+)([a-z0-9-]+)\/([^']*)'/g)) {
    const _up = m[1].length / 3
    const target = relative(lib, join(dirname(f), m[1], m[2])).split('/')[0]
    edges.push({ from: dom(f), file: rel(f), to: target, spec: m[0] })
  }
function dirname(p) {
  return p.slice(0, p.lastIndexOf('/'))
}
const known = new Set(readdirSync(lib).filter((e) => statSync(join(lib, e)).isDirectory()))
const bad = edges.filter((e) => known.has(e.to) && e.to !== e.from && !CORE.includes(e.to))
const against = edges.filter(
  (e) => known.has(e.to) && e.to !== e.from && CORE.includes(e.to) && ORDER.indexOf(e.to) > ORDER.indexOf(e.from),
)
out(
  `\nD. LAYERING: core files importing a constructed domain: ${bad.length}; core imports against the rank order (${ORDER.join(' < ')}): ${against.length}`,
)
const group = (xs) => {
  const g = {}
  for (const e of xs) {
    const k = `${e.from} -> ${e.to}`
    g[k] ??= new Set()
    g[k].add(e.file)
  }
  return Object.entries(g).sort((a, b) => b[1].size - a[1].size)
}
for (const [k, v] of group(bad)) out(`  ${k} (${v.size}): ${[...v].slice(0, 3).join(', ')}`)
out('  -- against the order:')
for (const [k, v] of group(against)) out(`  ${k} (${v.size}): ${[...v].slice(0, 3).join(', ')}`)

// ---- E. cast density ----
const dens = core
  .map((f) => {
    const t = text.get(f)
    const n = (t.match(/\sas (?!const\b)(?!\{)/g) ?? []).length
    const lines = t.split('\n').length
    const unk = (t.match(/as unknown as/g) ?? []).length
    return { f: rel(f), n, lines, unk, per100: +((100 * n) / lines).toFixed(1) }
  })
  .filter((x) => x.lines > 60)
  .sort((a, b) => b.per100 - a.per100)
out(
  `\nE. CAST DENSITY (the ' as T' assertions per 100 lines, core files over 60 lines): total ${dens.reduce((a, x) => a + x.n, 0)} casts, ${dens.reduce((a, x) => a + x.unk, 0)} 'as unknown as'`,
)
for (const x of dens.slice(0, 12))
  out(`  ${x.per100}/100 lines  ${x.n} casts (${x.unk} unknown) in ${x.lines} lines: ${x.f}`)
