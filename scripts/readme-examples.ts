/**
 * Run the programs in `examples/` and write what they make: the figure of each (`examples/<name>/figure.svg`), the
 * tapes of its runs (`tape.json`, or `tape-<name>.json` when there are several) and the generated index
 * `examples/README.md`.
 *
 *   node scripts/readme-examples.ts            run every example and write the files (or: make readme-examples)
 *   node scripts/readme-examples.ts <name>...  only these examples (the index is rewritten from all of them)
 *   node scripts/readme-examples.ts --check    run into a temporary directory and fail on any difference from the
 *                                              files in `examples/` (or: make readme-examples-check)
 *
 * An example is a folder `examples/<name>/main.tsx` that follows `examples/_shared/contract.ts`: `meta`, `run()` for the
 * compute and `View` for the figure. Nothing lists the examples: the folders are found here. Everything is seeded and
 * no clock is read, so a rerun writes identical bytes. The files run through tsx (for the JSX) with the `source`
 * export condition (the packages resolve to their TypeScript), so the script starts itself again with those flags when
 * it was started without them.
 */
import { spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const ROOT = join(import.meta.dirname, '..')
const EXAMPLES = join(ROOT, 'examples')

if (!process.execArgv.some((a) => a.includes('tsx'))) {
  const flags = ['--conditions=source', '--import', 'tsx', '--no-warnings']
  const again = spawnSync(process.execPath, [...flags, fileURLToPath(import.meta.url), ...process.argv.slice(2)], {
    stdio: 'inherit',
    cwd: ROOT,
  })
  process.exit(again.status ?? 1)
}

const args = process.argv.slice(2)
const check = args.includes('--check')
const only = new Set(check ? [] : args.filter((a) => !a.startsWith('--')))

const { renderView, tapeJson } = await import('../examples/_shared/render.ts')
const { prettyJson } = await import('../examples/_shared/pretty-json.ts')
type ExampleModule = import('../examples/_shared/contract.ts').ExampleModule
type Tape = import('../examples/_shared/contract.ts').Tape

const names = readdirSync(EXAMPLES, { withFileTypes: true })
  .filter((d) => d.isDirectory() && existsSync(join(EXAMPLES, d.name, 'main.tsx')))
  .map((d) => d.name)
  .sort()

/** The files an example makes, by path relative to `examples/`. */
const made = new Map<string, string>()
const entries: { name: string; module: ExampleModule; tapeFiles: string[] }[] = []

for (const name of names) {
  const module = (await import(pathToFileURL(join(EXAMPLES, name, 'main.tsx')).href)) as Partial<ExampleModule>
  if (module.meta === undefined || module.run === undefined || module.View === undefined)
    throw new Error(`examples/${name}/main.tsx must export meta, run and View (examples/_shared/contract.ts)`)
  const entry = { name, module: module as ExampleModule, tapeFiles: [] as string[] }
  entries.push(entry)
  if (only.size > 0 && !only.has(name)) continue
  console.log(name)
  const { result, tapes = {}, notes = [] } = await entry.module.run()
  for (const note of notes) console.log(`  ${note}`)
  const svg = renderView(entry.module.View, result)
  if (!svg.startsWith('<svg')) throw new Error(`examples/${name}: View must render an <svg> document`)
  made.set(`${name}/figure.svg`, `${svg}\n`)
  const keys = Object.keys(tapes)
  for (const key of keys) {
    const tape = tapes[key] as Tape
    const file = keys.length === 1 ? 'tape.json' : `tape-${key}.json`
    const json = await tapeJson(key, tape, entry.module.meta.title)
    made.set(`${name}/${file}`, `${prettyJson(json)}\n`)
    entry.tapeFiles.push(file)
  }
  console.log(`  ${(svg.length / 1024).toFixed(0)} KB figure, ${keys.length} tape(s)`)
}

/** The index of all the examples, README ones first. */
function indexOf(): string {
  const shown = entries.filter((e) => e.module.meta.readme)
  const more = entries.filter((e) => !e.module.meta.readme)
  const section = (e: (typeof entries)[number]): string => {
    const tapes = tapeFilesOf(e.name)
    return [
      `## ${e.module.meta.title}`,
      '',
      e.module.meta.summary,
      '',
      `![${e.module.meta.alt}](${e.name}/figure.svg)`,
      '',
      `[${e.name}/main.tsx](${e.name}/main.tsx)${tapes.map((t) => `, tape: [${t}](${e.name}/${t})`).join('')}`,
      '',
    ].join('\n')
  }
  return [
    '# Examples',
    '',
    'Each folder is a program. `main.tsx` computes with `mdml-engine` (a traced run, an estimator session) and draws the',
    'result with `mdml-render`; `figure.svg` is its figure and `tape.json` the tape of the run behind it, which the',
    "site's player can replay. `make readme-examples` runs them all and writes these files (`make readme-examples-check`",
    'fails when they are stale). To add an example, add a folder with a `main.tsx` that follows',
    '`_shared/contract.ts`; nothing else lists it. This file is generated.',
    '',
    ...(shown.length > 0 ? ['# In the README', '', ...shown.map(section)] : []),
    ...(more.length > 0 ? ['# More', '', ...more.map(section)] : []),
  ].join('\n')
}

/** The tape files of an example: the ones just made, or, when it was not run now, the ones on disk. */
function tapeFilesOf(name: string): string[] {
  const now = entries.find((e) => e.name === name)?.tapeFiles ?? []
  if (now.length > 0) return now
  const dir = join(EXAMPLES, name)
  return existsSync(dir)
    ? readdirSync(dir)
        .filter((f) => /^tape(-.*)?\.json$/.test(f))
        .sort()
    : []
}

made.set('README.md', `${indexOf()}\n`)

function write(root: string): void {
  for (const [path, text] of made) {
    mkdirSync(dirname(join(root, path)), { recursive: true })
    writeFileSync(join(root, path), text)
  }
}

if (!check) {
  write(EXAMPLES)
  console.log(`wrote ${made.size} files in examples/`)
} else {
  const tmp = mkdtempSync(join(tmpdir(), 'readme-examples-'))
  try {
    write(tmp)
    const differ: string[] = []
    for (const path of made.keys()) {
      const have = existsSync(join(EXAMPLES, path)) ? readFileSync(join(EXAMPLES, path), 'utf8') : undefined
      if (have !== readFileSync(join(tmp, path), 'utf8')) differ.push(path)
    }
    // Generated files left in an example's folder that nothing makes any more.
    for (const name of names)
      for (const f of readdirSync(join(EXAMPLES, name)))
        if (/^(figure\.svg|tape(-.*)?\.json)$/.test(f) && !made.has(`${name}/${f}`)) differ.push(`${name}/${f} (stale)`)
    if (differ.length > 0) {
      console.error(`readme-examples-check: ${differ.length} file(s) differ from a fresh run:`)
      for (const d of differ) console.error(`  examples/${d}`)
      console.error('run `make readme-examples` and commit the result')
      process.exit(1)
    }
    console.log(`readme-examples-check: ${made.size} files match a fresh run`)
  } finally {
    rmSync(tmp, { recursive: true, force: true })
  }
}
