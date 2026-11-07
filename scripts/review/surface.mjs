import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'

const R = join(import.meta.dirname, '..', '..')
const lines = readFileSync(process.argv[2], 'utf8')
  .split('\n')
  .filter(Boolean)
  .map((l) => l.split('\t'))
const CORE = new Set([
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
])
const exp = lines.map(([id, kind]) => {
  const [mod, name] = id.split('#')
  const parts = mod.split('/')
  return { id, kind, mod, name, pkg: parts[0], dom: parts[1] ?? '', sub: parts.slice(1).join('/') }
})
// ---- collisions: the same export name in several entry points of the engine ----
const byName = new Map()
for (const e of exp.filter((e) => e.pkg === 'mdml-engine'))
  (byName.get(e.name) ?? byName.set(e.name, []).get(e.name)).push(e)
const coll = [...byName.entries()].filter(([, v]) => new Set(v.map((e) => e.mod)).size > 1)
console.log(
  `COLLISIONS: ${coll.length} engine export names appear in 2+ entry points (of ${byName.size} distinct names)`,
)
const coreColl = coll.filter(([, v]) => v.some((e) => CORE.has(e.dom)))
console.log(`  with at least one core definition: ${coreColl.length}`)
for (const [n, v] of coreColl.sort((a, b) => b[1].length - a[1].length).slice(0, 14))
  console.log(
    `  ${n} x${v.length}: ${[...new Set(v.map((e) => e.mod.replace('mdml-engine/', '')))].slice(0, 6).join(', ')}`,
  )
// ---- exports of the core not mentioned in any other source file (word index over source, not tests) ----
const walk = (d, out = []) => {
  for (const e of readdirSync(d)) {
    if (['node_modules', 'dist', '.generated', 'index', 'fixtures'].includes(e)) continue
    const p = join(d, e)
    statSync(p).isDirectory() ? walk(p, out) : /\.tsx?$/.test(e) && out.push(p)
  }
  return out
}
const srcs = [...walk(`${R}/packages`), ...walk(`${R}/apps`)].filter(
  (f) =>
    !/\.(test|property\.test|golden\.test|test-d)\.tsx?$/.test(f) && !/\/index\.ts$/.test(f) && !/\.data\.ts$/.test(f),
)
const seen = new Map()
for (const f of srcs) {
  const ids = new Set(readFileSync(f, 'utf8').match(/[A-Za-z_]\w*/g) ?? [])
  for (const w of ids) (seen.get(w) ?? seen.set(w, new Set()).get(w)).add(f)
}
const unused = exp
  .filter((e) => e.pkg === 'mdml-engine' && CORE.has(e.dom))
  .filter((e) => {
    const fs = seen.get(e.name)
    return !fs || fs.size <= 1
  })
const total = exp.filter((e) => e.pkg === 'mdml-engine' && CORE.has(e.dom)).length
const perDom = {}
for (const e of unused) perDom[e.dom] = (perDom[e.dom] ?? 0) + 1
console.log(
  `\nCORE EXPORTS mentioned in at most their own source file (no other source file, site or poc uses them; tests do not count): ${unused.length} of ${total}`,
)
console.log(
  '  by domain: ' +
    Object.entries(perDom)
      .sort((a, b) => b[1] - a[1])
      .map(([d, n]) => `${d} ${n}`)
      .join(', '),
)
console.log(
  '  kinds: ' +
    Object.entries(Object.groupBy(unused, (e) => e.kind))
      .map(([k, v]) => [k, v.length])
      .map(([k, n]) => `${k} ${n}`)
      .join(', '),
)
