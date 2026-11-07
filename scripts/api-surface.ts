/**
 * The public surface of the packages as data, the guard for refactors that must not change it (docs/code-review/plan.md).
 *
 *   node scripts/api-surface.ts                 write docs/api-surface.txt: per entry point, its export count and a digest
 *   node scripts/api-surface.ts --check         fail, listing the entry points whose exports or signatures changed
 *   node scripts/api-surface.ts --full <file>   write every export (id, kind, signature digest) to a file
 *   node scripts/api-surface.ts --compare <file>  print the exports added, removed or changed against a `--full` file
 *
 * An export is `<module>#<name>` with its kind and a digest of its declared signature (every overload), as the docs
 * extractor reads it: a renamed, removed, added or retyped export changes the digest of its entry point. Doc comments
 * and bodies do not. Deterministic (sorted, no timestamps).
 */
import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

const ROOT = join(import.meta.dirname, '..')
const DIGEST = join(ROOT, 'docs', 'api-surface.txt')
const args = process.argv.slice(2)

const { readProject } = await import('../packages/meta/src/extract/project.ts')
const { readSymbols } = await import('../packages/meta/src/extract/symbols.ts')

const sha = (text: string): string => createHash('sha1').update(text).digest('hex').slice(0, 12)

/** Every export as `id<TAB>kind<TAB>signature digest`, sorted. */
function exportLines(): string[] {
  const project = readProject(ROOT, ['engine', 'render', 'meta'])
  const { symbols } = readSymbols(project)
  return symbols.map((s) => `${s.id}\t${s.kind}\t${sha(s.signature)}`).sort()
}

/** The per-entry-point digest lines: `module<TAB>count<TAB>digest`, then a total. */
function digestLines(lines: readonly string[]): string[] {
  const byModule = new Map<string, string[]>()
  for (const l of lines) {
    const module = (l.split('#')[0] ?? '').trim()
    byModule.set(module, [...(byModule.get(module) ?? []), l])
  }
  const out = [...byModule.entries()]
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([m, ls]) => `${m}\t${ls.length}\t${sha(ls.join('\n'))}`)
  out.push(`total\t${lines.length}\t${sha(lines.join('\n'))}`)
  return out
}

const lines = exportLines()
const full = args.indexOf('--full')
const compare = args.indexOf('--compare')
if (full >= 0) {
  writeFileSync(args[full + 1] as string, `${lines.join('\n')}\n`)
  console.log(`api-surface: ${lines.length} exports written`)
} else if (compare >= 0) {
  const before = new Map(
    readFileSync(args[compare + 1] as string, 'utf8')
      .split('\n')
      .filter((l) => l !== '')
      .map((l) => [l.split('\t')[0] as string, l] as const),
  )
  const after = new Map(lines.map((l) => [l.split('\t')[0] as string, l] as const))
  const added = [...after.keys()].filter((k) => !before.has(k))
  const removed = [...before.keys()].filter((k) => !after.has(k))
  const changed = [...after.keys()].filter((k) => before.has(k) && before.get(k) !== after.get(k))
  for (const k of added) console.log(`added    ${k}`)
  for (const k of removed) console.log(`removed  ${k}`)
  for (const k of changed) console.log(`changed  ${k}`)
  console.log(
    `api-surface: ${added.length} added, ${removed.length} removed, ${changed.length} changed of ${after.size}`,
  )
  if (added.length + removed.length + changed.length > 0) process.exit(1)
} else {
  const text = `${digestLines(lines).join('\n')}\n`
  if (args.includes('--check')) {
    const current = new Map(
      readFileSync(DIGEST, 'utf8')
        .split('\n')
        .filter((l) => l !== '')
        .map((l) => [l.split('\t')[0] as string, l] as const),
    )
    const now = new Map(digestLines(lines).map((l) => [l.split('\t')[0] as string, l] as const))
    const differs = [...new Set([...current.keys(), ...now.keys()])].filter((k) => current.get(k) !== now.get(k))
    if (differs.length > 0) {
      for (const k of differs) console.error(`api surface changed: ${k}`)
      console.error(
        'api-surface: the public surface differs from docs/api-surface.txt: run `make api-surface` if intended',
      )
      process.exit(1)
    }
    console.log(`api-surface: ${lines.length} exports, unchanged`)
  } else {
    writeFileSync(DIGEST, text)
    console.log(`api-surface: ${lines.length} exports in ${digestLines(lines).length - 1} entry points`)
  }
}
