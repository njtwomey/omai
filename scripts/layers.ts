/**
 * Check the layering rules of docs/structure.md section 2.
 *
 *   node scripts/layers.ts
 *
 * - A core domain (packages/engine/src/lib/<domain>) imports only domains of a lower tier (from its domain.json),
 *   or itself.
 * - The core never imports methods.
 * - A methods area (packages/engine/src/methods/<area>) imports any core domain, itself, and only the areas listed in
 *   its area.json `dependsOn`.
 * - whitebox-engine never imports whitebox-render or whitebox-meta; whitebox-render never imports whitebox-meta;
 *   headless render code never imports React or render/react.
 * Test files are exempt: tests may use higher layers to check lower ones.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'

const ROOT = join(import.meta.dirname, '..')
const ENGINE = join(ROOT, 'packages/engine/src')
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

function readJson(path: string): Record<string, unknown> {
  return existsSync(path) ? (JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>) : {}
}

const tiers = new Map<string, number>()
for (const domain of existsSync(join(ENGINE, 'lib')) ? readdirSync(join(ENGINE, 'lib')) : []) {
  const manifest = readJson(join(ENGINE, 'lib', domain, 'domain.json'))
  if (typeof manifest['tier'] === 'number') tiers.set(domain, manifest['tier'])
}
const areaDeps = new Map<string, readonly string[]>()
for (const area of existsSync(join(ENGINE, 'methods')) ? readdirSync(join(ENGINE, 'methods')) : []) {
  const manifest = readJson(join(ENGINE, 'methods', area, 'area.json'))
  areaDeps.set(area, Array.isArray(manifest['dependsOn']) ? (manifest['dependsOn'] as string[]) : [])
}

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
      if (from.kind === 'lib' && to.kind === 'lib' && from.unit !== to.unit) {
        const a = tiers.get(from.unit)
        const b = tiers.get(to.unit)
        if (a === undefined || b === undefined) errors.push(`${where}: domain without a tier in domain.json`)
        else if (b >= a) errors.push(`${where}: ${from.unit} (tier ${a}) may not import ${to.unit} (tier ${b})`)
      }
      if (from.kind === 'methods' && to.kind === 'methods' && from.unit !== to.unit) {
        if (!(areaDeps.get(from.unit) ?? []).includes(to.unit)) {
          errors.push(`${where}: add '${to.unit}' to methods/${from.unit}/area.json dependsOn`)
        }
      }
      continue
    }
    if (from.package === 'engine' && /^whitebox-(render|meta)/.test(spec))
      errors.push(`${where}: engine is the base package`)
    if (from.package === 'render' && spec.startsWith('whitebox-meta'))
      errors.push(`${where}: render may not import meta`)
    if (from.package === 'render' && from.unit !== 'react' && /^(react|react-dom)(\/|$)/.test(spec)) {
      errors.push(`${where}: headless render code may not import React`)
    }
  }
}

if (errors.length > 0) {
  for (const e of errors) console.error(e)
  console.error(`${errors.length} layering violation(s)`)
  process.exit(1)
}
console.log(`layers: ${files.length} files checked, ${tiers.size} domains, ${areaDeps.size} areas`)
