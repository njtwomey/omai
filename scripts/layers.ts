/**
 * Check the layering rules of docs/structure.md section 2.
 *
 *   node scripts/layers.ts
 *
 * - The core domains (packages/engine/src/lib/<domain>) import one another without a cycle: if `stats` imports
 *   `dists`, nothing `dists` imports may lead back to `stats`. A cycle is reported as its path, with one import for
 *   each step.
 * - The core never imports methods.
 * - The methods areas (packages/engine/src/methods/<area>) import any core domain and one another, again without a
 *   cycle.
 * - mdml-engine never imports mdml-render or mdml-meta; mdml-render never imports mdml-meta or engine methods (it
 *   draws core contracts and exposition records, never algorithms);
 *   headless render code never imports React or render/react.
 * Test files are exempt: tests may use higher layers to check lower ones.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'

const ROOT = join(import.meta.dirname, '..')
const IMPORT = /(?:^|\n)\s*(?:import|export)\s[^'"]*?from\s+['"]([^'"]+)['"]|import\(\s*['"]([^'"]+)['"]\s*\)/g

interface Location {
  readonly package: string
  readonly kind: 'lib' | 'methods' | 'other'
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
  if (pkg === 'engine' && rel[2] === 'methods') return { package: pkg, kind: 'methods', unit: rel[3] ?? '' }
  return { package: pkg, kind: 'other', unit: rel[2] ?? '' }
}

/** The import graph between units (core domains, or methods areas): one example import per edge. */
type Graph = Map<string, Map<string, string>>

function addEdge(graph: Graph, from: string, to: string, where: string): void {
  const out = graph.get(from) ?? new Map<string, string>()
  if (!out.has(to)) out.set(to, where)
  graph.set(from, out)
}

/** Every cycle of a graph, one per strongly connected component (Tarjan), as a path that returns to its start. */
function cycles(graph: Graph): string[][] {
  const index = new Map<string, number>()
  const low = new Map<string, number>()
  const stack: string[] = []
  const onStack = new Set<string>()
  const found: string[][] = []
  let next = 0
  const connect = (v: string): void => {
    index.set(v, next)
    low.set(v, next)
    next++
    stack.push(v)
    onStack.add(v)
    for (const w of (graph.get(v) ?? new Map()).keys()) {
      if (!index.has(w)) {
        connect(w)
        low.set(v, Math.min(low.get(v) as number, low.get(w) as number))
      } else if (onStack.has(w)) low.set(v, Math.min(low.get(v) as number, index.get(w) as number))
    }
    if (low.get(v) !== index.get(v)) return
    const component = new Set<string>()
    let w: string
    do {
      w = stack.pop() as string
      onStack.delete(w)
      component.add(w)
    } while (w !== v)
    if (component.size > 1) found.push(shortestCycle(graph, v, component))
  }
  for (const v of [...graph.keys()].sort()) if (!index.has(v)) connect(v)
  return found
}

/** The shortest path from `start` back to itself inside a strongly connected component (breadth first). */
function shortestCycle(graph: Graph, start: string, component: ReadonlySet<string>): string[] {
  const previous = new Map<string, string>()
  const queue = [start]
  while (queue.length > 0) {
    const v = queue.shift() as string
    for (const w of [...(graph.get(v) ?? new Map()).keys()].sort()) {
      if (!component.has(w)) continue
      if (w === start) {
        const path: string[] = []
        for (let u = v; u !== start; u = previous.get(u) as string) path.unshift(u)
        return [start, ...path, start]
      }
      if (!previous.has(w)) {
        previous.set(w, v)
        queue.push(w)
      }
    }
  }
  return [start, start]
}

const domains: Graph = new Map()
const areas: Graph = new Map()
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
      if (from.kind === 'lib' && to.kind === 'methods') errors.push(`${where}: the core never imports methods`)
      if (from.kind === 'lib' && to.kind === 'lib' && from.unit !== to.unit) addEdge(domains, from.unit, to.unit, where)
      if (from.kind === 'methods' && to.kind === 'methods' && from.unit !== to.unit)
        addEdge(areas, from.unit, to.unit, where)
      continue
    }
    if (from.package === 'engine' && /^mdml-(render|meta)/.test(spec))
      errors.push(`${where}: engine is the base package`)
    if (from.package === 'render' && spec.startsWith('mdml-meta')) errors.push(`${where}: render may not import meta`)
    if (from.package === 'render' && spec.startsWith('mdml-engine/methods'))
      errors.push(`${where}: render draws contracts and exposition records, never methods`)
    if (from.package === 'render' && from.unit !== 'react' && /^(react|react-dom)(\/|$)/.test(spec)) {
      errors.push(`${where}: headless render code may not import React`)
    }
  }
}

for (const [graph, kind] of [
  [domains, 'core domains'],
  [areas, 'methods areas'],
] as const) {
  for (const cycle of cycles(graph)) {
    const steps = cycle.slice(1).map((to, i) => `  ${graph.get(cycle[i] as string)?.get(to)}`)
    errors.push(`import cycle between ${kind}: ${cycle.join(' -> ')}\n${steps.join('\n')}`)
  }
}

if (errors.length > 0) {
  for (const e of errors) console.error(e)
  console.error(`${errors.length} layering violation(s)`)
  process.exit(1)
}
const units = (g: Graph): number => new Set([...g.keys(), ...[...g.values()].flatMap((m) => [...m.keys()])]).size
console.log(
  `layers: ${files.length} files checked; ${units(domains)} domains and ${units(areas)} areas import without cycles`,
)
