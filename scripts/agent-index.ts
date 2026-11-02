/**
 * Generate the agent index of the engine library from its `@index` tags (docs/structure.md section 7).
 *
 *   node scripts/agent-index.ts          write packages/engine/llms.txt and packages/engine/index/
 *   node scripts/agent-index.ts --check  fail with the list of stale files, or on any tag error
 *
 * The source of truth is the TSDoc: `@index <kind> <description>` in a file's opening paragraph or on a declaration.
 * The usage note at the top of `llms.txt` is hand-written in packages/engine/llms.header.md. Tag errors (unknown kind,
 * empty or over-long description, duplicate tag or key) print as `file:line: message` and fail the run; warnings print
 * and do not. The outputs are committed and deterministic (sorted, no timestamps).
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

const ROOT = join(import.meta.dirname, '..')
const PACKAGE = join(ROOT, 'packages/engine')
const check = process.argv.includes('--check')

// Loaded here, not at the top: ts-morph takes a moment to import.
const { AGENT_INDEX_LAYOUT, buildAgentIndex, staleAgentIndex } = await import(
  '../packages/meta/src/extract/agent-index.ts'
)
const { collected, files } = buildAgentIndex(ROOT)

let errors = 0
for (const p of collected.problems) {
  if (p.severity === 'error') errors++
  console.error(`${p.file}:${p.line}: ${p.severity}: ${p.message}`)
}

if (errors > 0) {
  console.error(`agent-index: ${errors} tag error(s)`)
  process.exit(1)
}

/** The generated files already on disk, by path relative to the package: llms.txt and everything under index/. */
function existing(): Map<string, string> {
  const out = new Map<string, string>()
  const read = (path: string): void => {
    out.set(path, readFileSync(join(PACKAGE, path), 'utf8'))
  }
  if (existsSync(join(PACKAGE, AGENT_INDEX_LAYOUT.llms))) read(AGENT_INDEX_LAYOUT.llms)
  const dir = join(PACKAGE, AGENT_INDEX_LAYOUT.dir)
  if (existsSync(dir)) for (const name of readdirSync(dir)) read(`${AGENT_INDEX_LAYOUT.dir}/${name}`)
  return out
}

const stale = staleAgentIndex(files, existing())
if (!check) {
  for (const path of stale) {
    const full = join(PACKAGE, path.replace(/ \(no longer generated\)$/, ''))
    const content = files.get(path)
    if (content === undefined) rmSync(full)
    else {
      mkdirSync(dirname(full), { recursive: true })
      writeFileSync(full, content)
    }
  }
}

if (check && stale.length > 0) {
  for (const s of stale) console.error(`stale agent index: packages/engine/${s}`)
  console.error(`${stale.length} agent index file(s) out of date: run make agent-index`)
  process.exit(1)
}
if (!check)
  console.log(`agent-index: ${collected.entries.length} entries, ${files.size} files, ${stale.length} updated`)
