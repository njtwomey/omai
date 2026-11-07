/**
 * Check the layering rules of docs/structure.md section 2.
 *
 *   node scripts/layers.ts
 *
 * - The core domains (packages/engine/src/lib/<domain>) import one another without a cycle: if `stats` imports
 *   `dists`, nothing `dists` imports may lead back to `stats`. A cycle is reported as its path, with one import for
 *   each step.
 * - The tiers (scripts/layers-rules.ts): the foundation domains import only the short list of domains below them, and
 *   the machinery (numerics, dists, optim, losses, kernels, stats, metrics, information, data) imports only the
 *   foundation and itself, except for the few imports listed there with their reason. No total order.
 * - No file-level import cycle inside a domain (value imports only), except the ones written out in the allowlist of
 *   scripts/layers-rules.ts; the cycles found are printed.
 * - mdml-engine never imports mdml-render or mdml-meta; mdml-render never imports mdml-meta (it draws core contracts
 *   and exposition records, never algorithms); headless render code never imports React or render/react.
 * Test files are exempt: tests may use higher layers to check lower ones.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import {
  addEdge,
  cycles,
  fileCycleReport,
  type Graph,
  type ImportEdge,
  staleExceptions,
  tierViolations,
} from './layers-rules.ts'

const ROOT = join(import.meta.dirname, '..')
const IMPORT = /(?:^|\n)\s*(?:import|export)\s[^'"]*?from\s+['"]([^'"]+)['"]|import\(\s*['"]([^'"]+)['"]\s*\)/g
const TYPE_ONLY = /^\s*(?:import|export)\s+type[\s{]/
const LIB = join(ROOT, 'packages', 'engine', 'src', 'lib')

interface Location {
  readonly package: string
  readonly kind: 'lib' | 'other'
  readonly unit: string
}

function walk(dir: string, out: string[] = []): string[] {
  if (!existsSync(dir)) return out
  for (const name of readdirSync(dir)) {
    const path = join(dir, name)
    if (statSync(path).isDirectory()) walk(path, out)
    else if (
      /\.tsx?$/.test(name) &&
      !/\.test(-d)?\.tsx?$/.test(name) &&
      !/\.(property|golden|browser)\.test\.tsx?$/.test(name)
    ) {
      out.push(path)
    }
  }
  return out
}

function locate(path: string): Location {
  const rel = relative(join(ROOT, 'packages'), path).split('/')
  const pkg = rel[0] ?? ''
  if (pkg === 'engine' && rel[2] === 'lib') return { package: pkg, kind: 'lib', unit: rel[3] ?? '' }
  return { package: pkg, kind: 'other', unit: rel[2] ?? '' }
}

/** The file a relative import names (the specifier carries its `.ts`; a folder or a bare name is tried as well). */
function targetFile(from: string, spec: string): string {
  const base = resolve(dirname(from), spec)
  for (const candidate of [base, `${base}.ts`, join(base, 'index.ts')]) {
    if (existsSync(candidate) && statSync(candidate).isFile()) return candidate
  }
  return base
}

const domains: Graph = new Map()
const edges: ImportEdge[] = []
const errors: string[] = []
const files = ['engine', 'render', 'meta'].flatMap((p) => walk(join(ROOT, 'packages', p, 'src')))
for (const file of files) {
  const from = locate(file)
  const source = readFileSync(file, 'utf8')
  for (const match of source.matchAll(IMPORT)) {
    const spec = match[1] ?? match[2] ?? ''
    const where = `${relative(ROOT, file)} imports '${spec}'`
    if (spec.startsWith('.')) {
      const to = locate(resolve(dirname(file), spec))
      if (from.package !== 'engine' || to.package !== 'engine') continue
      if (from.kind === 'lib' && to.kind === 'lib' && from.unit !== to.unit) addEdge(domains, from.unit, to.unit, where)
      if (from.kind === 'lib' && to.kind === 'lib') {
        edges.push({
          from: relative(LIB, file),
          to: relative(LIB, targetFile(file, spec)),
          typeOnly: TYPE_ONLY.test(match[0]),
          where,
        })
      }
      continue
    }
    if (from.package === 'engine' && /^mdml-(render|meta)/.test(spec))
      errors.push(`${where}: engine is the base package`)
    if (from.package === 'render' && spec.startsWith('mdml-meta')) errors.push(`${where}: render may not import meta`)
    if (from.package === 'render' && from.unit !== 'react' && /^(react|react-dom)(\/|$)/.test(spec)) {
      errors.push(`${where}: headless render code may not import React`)
    }
  }
}

for (const { path } of cycles(domains)) {
  const steps = path.slice(1).map((to, i) => `  ${domains.get(path[i] as string)?.get(to)}`)
  errors.push(`import cycle between core domains: ${path.join(' -> ')}\n${steps.join('\n')}`)
}

errors.push(...tierViolations(edges), ...staleExceptions(edges))

const fileCycles = fileCycleReport(edges)
for (const c of fileCycles.found) {
  const steps = c.path.slice(1).map((to, i) => `  ${c.path[i]} -> ${to}`)
  const text = `${c.members.length} files, shortest cycle:\n${steps.join('\n')}`
  if (c.allowed === null) errors.push(`file-level import cycle (new): ${text}`)
  else console.log(`layers: allowed file cycle, ${text}\n  because ${c.allowed.reason}`)
}
errors.push(...fileCycles.stale)

if (errors.length > 0) {
  for (const e of errors) console.error(e)
  console.error(`${errors.length} layering violation(s)`)
  process.exit(1)
}
const units = (g: Graph): number => new Set([...g.keys(), ...[...g.values()].flatMap((m) => [...m.keys()])]).size
console.log(
  `layers: ${files.length} files checked; ${units(domains)} domains import without cycles; ${fileCycles.found.length} allowed file cycle(s)`,
)
