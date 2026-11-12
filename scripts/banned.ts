/**
 * Fail on banned globals in published source: Math.random (use keys from omai-engine/random), Date.now and
 * performance.now in algorithms (traces must be deterministic), and console output (use trace events). The engine, which
 * is deterministic by contract, also bans the other clocks and sources of entropy: `new Date(`, `crypto.getRandomValues`,
 * `crypto.randomUUID`, `process.hrtime` and timers (`setTimeout`, `setInterval`, `requestAnimationFrame`).
 *
 *   node scripts/banned.ts
 *
 * Tests, scripts and apps are exempt. A line may opt out with a trailing `// allow: <reason>` comment.
 *
 * A second check (`scripts/banned-arrays.ts`) fails on hand-built numeric arrays (`new Array`, `.fill(`, `Array.from({`,
 * typed arrays) in the EP code paths and on hand loops in the code that pages show; a marker
 * `// banned-ok: <category>: <reason>` on the line above allows an occurrence that is not numeric data.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { arrayProblems, SHOWN_ROOTS } from './banned-arrays.ts'

const ROOT = join(import.meta.dirname, '..')
const RULES: readonly { readonly pattern: RegExp; readonly message: string; readonly engineOnly?: true }[] = [
  { pattern: /\bMath\.random\s*\(/, message: 'use keys from omai-engine/random' },
  { pattern: /\b(Date\.now|performance\.now)\s*\(/, message: 'clocks make traces non-deterministic' },
  { pattern: /\bconsole\.(log|info|warn|error|debug)\s*\(/, message: 'emit trace events instead of logging' },
  { pattern: /\bnew Date\s*\(/, message: 'clocks make traces non-deterministic', engineOnly: true },
  {
    pattern: /\bcrypto\.(getRandomValues|randomUUID)\b/,
    message: 'use keys from omai-engine/random',
    engineOnly: true,
  },
  { pattern: /\bprocess\.hrtime\b/, message: 'clocks make traces non-deterministic', engineOnly: true },
  {
    pattern: /\b(setTimeout|setInterval|requestAnimationFrame)\s*\(/,
    message: 'timers make runs depend on the clock',
    engineOnly: true,
  },
]

function walk(dir: string, out: string[] = []): string[] {
  if (!existsSync(dir)) return out
  for (const name of readdirSync(dir)) {
    const path = join(dir, name)
    if (statSync(path).isDirectory()) walk(path, out)
    else if (/\.tsx?$/.test(name) && !/\.test(-d)?\.tsx?$/.test(name)) out.push(path)
  }
  return out
}

const problems: string[] = []
for (const pkg of ['engine', 'render', 'meta']) {
  for (const file of walk(join(ROOT, 'packages', pkg, 'src'))) {
    if (file.includes('/testing/')) continue
    readFileSync(file, 'utf8')
      .split('\n')
      .forEach((line, i) => {
        if (line.includes('// allow:')) return
        for (const rule of RULES) {
          if (rule.engineOnly === true && pkg !== 'engine') continue
          if (rule.pattern.test(line)) problems.push(`${relative(ROOT, file)}:${i + 1}: ${rule.message}`)
        }
      })
  }
}
// Hand-built numeric arrays: the files of the EP code paths and of the code pages show (tests included in the walk, skipped by the rule).
const sourcesUnder = (dir: string, out: string[] = []): string[] => {
  if (!existsSync(dir)) return out
  for (const name of readdirSync(dir)) {
    const path = join(dir, name)
    if (name === 'node_modules' || name.startsWith('.')) continue
    if (statSync(path).isDirectory()) sourcesUnder(path, out)
    else if (/\.tsx?$/.test(name)) out.push(path)
  }
  return out
}
for (const root of SHOWN_ROOTS) {
  for (const file of sourcesUnder(join(ROOT, root))) {
    const path = relative(ROOT, file)
    for (const p of arrayProblems(path, readFileSync(file, 'utf8'))) problems.push(`${path}:${p.line}: ${p.message}`)
  }
}
if (problems.length > 0) {
  for (const p of problems) console.error(p)
  process.exit(1)
}
console.log('banned: clean')
