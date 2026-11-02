/**
 * Generate the agent index of the indexed packages from their `@index` tags (docs/structure.md section 7).
 *
 *   node scripts/agent-index.ts          write llms.txt, packages/<package>/llms.txt and packages/<package>/index/
 *   node scripts/agent-index.ts --check  fail with the list of stale files, or on any tag error
 *
 * The packages are `AGENT_INDEX_PACKAGES` (the engine and render). The source of truth is the TSDoc: `@index <kind>
 * <description>` in a file's opening paragraph or on a declaration. The usage note at the top of each package's
 * `llms.txt` is hand-written in `packages/<package>/llms.header.md`; the repository-level `llms.txt` is a generated map
 * of the packages. Tag errors (unknown kind, empty or over-long description, duplicate tag or key) print as
 * `file:line: message` and fail the run; warnings print and do not. The outputs are committed and deterministic
 * (sorted, no timestamps).
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

const ROOT = join(import.meta.dirname, '..')
const check = process.argv.includes('--check')

// Loaded here, not at the top: ts-morph takes a moment to import.
const { AGENT_INDEX_LAYOUT, AGENT_INDEX_ROOT_FILE, buildRepositoryIndex, staleAgentIndex } = await import(
  '../packages/meta/src/extract/agent-index.ts'
)
const built = buildRepositoryIndex(ROOT)

let errors = 0
let entries = 0
for (const b of built.packages) {
  entries += b.collected.entries.length
  for (const p of b.collected.problems) {
    if (p.severity === 'error') errors++
    console.error(`${p.file}:${p.line}: ${p.severity}: ${p.message}`)
  }
}
if (errors > 0) {
  console.error(`agent-index: ${errors} tag error(s)`)
  process.exit(1)
}

/** The generated files of a package already on disk, by path relative to it: llms.txt and everything under index/. */
function existing(packageDir: string): Map<string, string> {
  const out = new Map<string, string>()
  const read = (path: string): void => {
    out.set(path, readFileSync(join(packageDir, path), 'utf8'))
  }
  if (existsSync(join(packageDir, AGENT_INDEX_LAYOUT.llms))) read(AGENT_INDEX_LAYOUT.llms)
  const dir = join(packageDir, AGENT_INDEX_LAYOUT.dir)
  if (existsSync(dir)) for (const name of readdirSync(dir)) read(`${AGENT_INDEX_LAYOUT.dir}/${name}`)
  return out
}

/** The stale paths of one set of files, as repository-relative paths (an orphan keeps its note). */
function reconcile(
  prefix: string,
  packageDir: string,
  files: ReadonlyMap<string, string>,
  current: Map<string, string>,
): string[] {
  const stale = staleAgentIndex(files, current)
  if (!check) {
    for (const path of stale) {
      const full = join(packageDir, path.replace(/ \(no longer generated\)$/, ''))
      const content = files.get(path)
      if (content === undefined) rmSync(full)
      else {
        mkdirSync(dirname(full), { recursive: true })
        writeFileSync(full, content)
      }
    }
  }
  return stale.map((s) => `${prefix}${s}`)
}

const stale: string[] = []
for (const b of built.packages) {
  const dir = join(ROOT, 'packages', b.package.package)
  stale.push(...reconcile(`packages/${b.package.package}/`, dir, b.files, existing(dir)))
}
const rootFiles = new Map([[AGENT_INDEX_ROOT_FILE, built.root]])
const rootExisting = new Map<string, string>()
if (existsSync(join(ROOT, AGENT_INDEX_ROOT_FILE)))
  rootExisting.set(AGENT_INDEX_ROOT_FILE, readFileSync(join(ROOT, AGENT_INDEX_ROOT_FILE), 'utf8'))
stale.push(...reconcile('', ROOT, rootFiles, rootExisting))

if (check && stale.length > 0) {
  for (const s of stale) console.error(`stale agent index: ${s}`)
  console.error(`${stale.length} agent index file(s) out of date: run make agent-index`)
  process.exit(1)
}
if (!check) {
  const files = built.packages.reduce((n, b) => n + b.files.size, rootFiles.size)
  console.log(
    `agent-index: ${entries} entries in ${built.packages.length} packages, ${files} files, ${stale.length} updated`,
  )
}
