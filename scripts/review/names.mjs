/**
 * Names the function definitions of the engine, clusters them by name, and verifies each cluster by comparing bodies.
 *
 *   node scripts/review/names.mjs <report file> [similarity = 0.6] [minimum tokens = 12]
 */
import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync } from 'node:fs'
import { CORE, domainOf, engineSource, overlap, relOf, shingles, tokens } from './_shared.mjs'

const files = engineSource()
const dom = domainOf
const rel = relOf
// ---- definitions: top-level functions and arrow/const functions ----
const defs = []
for (const f of files) {
  const lines = readFileSync(f, 'utf8').split('\n')
  for (let i = 0; i < lines.length; i++) {
    const m =
      /^(export )?(?:async )?function\*? ([A-Za-z0-9_]+)|^(export )?const ([A-Za-z0-9_]+)(?:: [^=]+)? = (?:async )?(?:\([^)]*\)|[A-Za-z_]\w*)(?:: [^=]+)? =>/.exec(
        lines[i],
      )
    if (!m) continue
    const name = m[2] ?? m[4]
    const exported = Boolean(m[1] ?? m[3])
    let j = i
    if (!/[;}]\s*$/.test(lines[i]) || /\{\s*$/.test(lines[i])) {
      j = i + 1
      while (j < lines.length && !/^[})]/.test(lines[j])) j++
    }
    const body = lines.slice(i, j + 1).join('\n')
    const t = tokens(body.slice(Math.max(0, body.indexOf('{'))))
    if (t.length < 12) continue
    defs.push({
      name,
      exported,
      file: f,
      line: i + 1,
      len: j - i + 1,
      ntok: t.length,
      sh: shingles(t),
      hash: createHash('md5').update(t.join(' ')).digest('hex'),
    })
  }
}
// ---- name normalisation and clustering ----
const NOISE = new Set(['of', 'impl', 'helper', 'fn', 'internal', 'raw', 'fast', 'simple', 'the', 'with', 'to', 'for'])
const words = (n) =>
  n
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/_/g, ' ')
    .toLowerCase()
    .split(' ')
    .filter((w) => w && !NOISE.has(w))
    .map((w) => (w.length > 3 && w.endsWith('s') ? w.slice(0, -1) : w))
defs.forEach((d) => {
  d.w = words(d.name)
  d.key = [...d.w].sort().join('-')
})
const byKey = new Map()
for (const d of defs) (byKey.get(d.key) ?? byKey.set(d.key, []).get(d.key)).push(d)
// near names: two defs sharing a rare word, with Jaccard of word sets >= 0.67 (2+ words), joined into the exact-key clusters by union-find
const parent = new Map()
const find = (k) => {
  while (parent.get(k) !== k) {
    parent.set(k, parent.get(parent.get(k)))
    k = parent.get(k)
  }
  return k
}
for (const k of byKey.keys()) parent.set(k, k)
const keysByWord = new Map()
for (const k of byKey.keys()) for (const w of k.split('-')) (keysByWord.get(w) ?? keysByWord.set(w, []).get(w)).push(k)
for (const [, ks] of keysByWord) {
  if (ks.length > 60) continue
  for (let a = 0; a < ks.length; a++)
    for (let b = a + 1; b < ks.length; b++) {
      const A = new Set(ks[a].split('-')),
        B = new Set(ks[b].split('-'))
      if (A.size < 2 || B.size < 2) continue
      const inter = [...A].filter((x) => B.has(x)).length
      if (inter / (A.size + B.size - inter) >= 0.67) parent.set(find(ks[a]), find(ks[b]))
    }
}
const clusters = new Map()
for (const [k, ds] of byKey) {
  const r = find(k)
  ;(clusters.get(r) ?? clusters.set(r, []).get(r)).push(...ds)
}
// ---- verify: body similarity inside clusters with 2+ files ----
const sim = (a, b) => overlap(a.sh, b.sh)
const SIM = Number(process.argv[3] ?? 0.6),
  MINTOK = Number(process.argv[4] ?? 12)
const verified = []
for (const [, ds] of clusters) {
  const nf = new Set(ds.map((d) => d.file)).size
  if (nf < 2 || ds.length > 80) continue
  // group members whose bodies are similar (>= 0.6) to each other: connected components
  const comp = ds.map((_, i) => i)
  const fc = (i) => {
    while (comp[i] !== i) {
      comp[i] = comp[comp[i]]
      i = comp[i]
    }
    return i
  }
  for (let a = 0; a < ds.length; a++)
    for (let b = a + 1; b < ds.length; b++)
      if (ds[a].file !== ds[b].file && (ds[a].hash === ds[b].hash || sim(ds[a], ds[b]) >= SIM)) comp[fc(a)] = fc(b)
  const groups = new Map()
  for (const [i, d] of ds.entries()) (groups.get(fc(i)) ?? groups.set(fc(i), []).get(fc(i))).push(d)
  for (const g of groups.values()) {
    const files = new Set(g.map((d) => d.file))
    if (files.size >= 2 && Math.max(...g.map((d) => d.ntok)) >= MINTOK)
      verified.push({
        name: g[0].name,
        members: g,
        identical: new Set(g.map((d) => d.hash)).size === 1,
        files: files.size,
        core: g.some((d) => d.exported && CORE.has(dom(d.file))),
        outside: g.some((d) => !CORE.has(dom(d.file))),
      })
  }
}
verified.sort(
  (a, b) => b.files * Math.max(...b.members.map((d) => d.len)) - a.files * Math.max(...a.members.map((d) => d.len)),
)
const lines = []
const out = (s = '') => {
  lines.push(s)
}
out(
  `DEFINITIONS: ${defs.length} functions (>=12 tokens) in ${files.length} files; NAME CLUSTERS: ${clusters.size}; of them with 2+ files: ${[...clusters.values()].filter((c) => new Set(c.map((d) => d.file)).size >= 2).length}`,
)
out(
  `VERIFIED CLUSTERS (2+ files with bodies >= 60% alike): ${verified.length}; identical bodies: ${verified.filter((v) => v.identical).length}; with a core export copied outside the core: ${verified.filter((v) => v.core && v.outside).length}`,
)
const fmt = (v) =>
  `${v.identical ? 'IDENTICAL' : 'similar  '} x${v.members.length} in ${v.files} files: ${v.name} [${[...new Set(v.members.map((d) => d.name))].slice(0, 4).join('/')}] ` +
  v.members
    .slice(0, 5)
    .map((d) => `${rel(d.file)}:${d.line}(${d.len}${d.exported ? ',exp' : ''})`)
    .join(' ') +
  (v.members.length > 5 ? ' ...' : '')
out('\n-- core export re-implemented outside the core (verified):')
for (const v of verified.filter((v) => v.core && v.outside)) out(`  ${fmt(v)}`)
out('\n-- other verified clusters, largest first:')
for (const v of verified.filter((v) => !(v.core && v.outside))) out(`  ${fmt(v)}`)
writeFileSync(process.argv[2], `${lines.join('\n')}\n`)
const head = lines.slice(0, 2).join('\n')
console.log(head)
console.log('core-copied (first 25):')
console.log(lines.slice(3, 28).join('\n').slice(0, 7000))
