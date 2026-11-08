/**
 * Reads a full surface file (`node scripts/api-surface.ts --full <file>`): export names that appear in several entry
 * points, and core exports that no other source file mentions.
 *
 *   node scripts/review/surface.mjs <full surface file>
 */
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { CORE, listFor, ROOT, walk } from './_shared.mjs'

const exports = readFileSync(process.argv[2], 'utf8')
  .split('\n')
  .filter(Boolean)
  .map((line) => {
    const [id, kind] = line.split('\t')
    const [module, name] = id.split('#')
    const [pkg, domain = ''] = module.split('/')
    return { kind, module, name, pkg, domain }
  })
const engine = exports.filter((e) => e.pkg === 'omai-engine')
const inCore = engine.filter((e) => CORE.has(e.domain))

const byName = new Map()
for (const e of engine) listFor(byName, e.name).push(e)
const shared = [...byName].filter(([, v]) => new Set(v.map((e) => e.module)).size > 1)
const sharedCore = shared.filter(([, v]) => v.some((e) => CORE.has(e.domain)))
console.log(`COLLISIONS: ${shared.length} engine export names appear in 2+ entry points (of ${byName.size} distinct)`)
console.log(`  with at least one core definition: ${sharedCore.length}`)
for (const [name, v] of sharedCore.sort((a, b) => b[1].length - a[1].length).slice(0, 14)) {
  const modules = [...new Set(v.map((e) => e.module.replace('omai-engine/', '')))]
  console.log(`  ${name} x${v.length}: ${modules.slice(0, 6).join(', ')}`)
}

const mentions = new Map()
const sources = [...walk(join(ROOT, 'packages')), ...walk(join(ROOT, 'apps'))].filter(
  (f) =>
    !/\.(test|property\.test|golden\.test|test-d)\.tsx?$/.test(f) && !/\/index\.ts$/.test(f) && !/\.data\.ts$/.test(f),
)
for (const f of sources) {
  for (const word of new Set(readFileSync(f, 'utf8').match(/[A-Za-z_]\w*/g) ?? [])) listFor(mentions, word).push(f)
}
const unused = inCore.filter((e) => (mentions.get(e.name)?.length ?? 0) <= 1)
const count = (list, key) => {
  const counts = new Map()
  for (const e of list) counts.set(e[key], (counts.get(e[key]) ?? 0) + 1)
  return [...counts].sort((a, b) => b[1] - a[1]).map(([k, n]) => `${k} ${n}`)
}
console.log(
  `\nCORE EXPORTS mentioned in at most their own source file (tests do not count): ${unused.length} of ${inCore.length}`,
)
console.log(`  by domain: ${count(unused, 'domain').join(', ')}`)
console.log(`  kinds: ${count(unused, 'kind').join(', ')}`)
