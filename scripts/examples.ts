/**
 * Run every `@example` of whitebox-engine's exports and check each `// =>` line (docs/structure.md section 2).
 *
 *   node scripts/examples.ts [--only <prefix>[,<prefix>...]] [--methods] [--workers N] [--timeout ms] [--json file]
 *                            [--slowest N] [--quiet] [--coverage]
 *
 * The catalogue comes from whitebox-meta/extract and the examples run through whitebox-meta's runner in worker
 * threads (default 2, or WHITEBOX_EXAMPLE_WORKERS), each importing the engine from source. An example that runs past
 * the time limit (default 20 s) has its worker stopped and replaced. The core (`lib/`) is strict: any failure exits 1.
 * `--methods` also runs the methods' examples, which are optional and reported without failing.
 * `--only` keeps examples whose module (`whitebox-engine/numerics/linalg`) or file (`packages/engine/src/lib/stats`)
 * starts with one of the prefixes. `--coverage` also reports documentation coverage of the core and the
 * folders and files of every package without a summary (whitebox-meta/extract).
 */
import { availableParallelism } from 'node:os'
import { join } from 'node:path'
import { isMainThread, parentPort, Worker } from 'node:worker_threads'

const ROOT = join(import.meta.dirname, '..')

interface Job {
  readonly index: number
  readonly symbol: string
  readonly module: string
  readonly strict: boolean
  readonly file: string
  readonly line: number
  readonly code: string
}

interface JobResult {
  readonly index: number
  readonly ok: boolean
  readonly ms: number
  /** One line per problem: an error that stopped the example, or a failed expectation. */
  readonly problems: readonly string[]
}

type ToWorker = { readonly jobs: readonly Job[] } | { readonly done: true }
type FromWorker = { readonly ready: true } | { readonly start: number } | { readonly result: JobResult }

// Node flags the workers need: the `source` export condition (packages resolve to their TypeScript) and type
// transformation (a few engine files use parameter properties).
const WORKER_FLAGS = ['--conditions=source', '--experimental-transform-types', '--no-warnings']

async function worker(): Promise<void> {
  const port = parentPort
  if (port === null) return
  process.on('unhandledRejection', () => {})
  const { runExample } = await import('../packages/meta/src/runner/index.ts')
  const { moduleLoaders } = await import('../packages/meta/src/modules/index.ts')
  port.on('message', async (msg: ToWorker) => {
    if ('done' in msg) {
      port.close()
      return
    }
    for (const job of msg.jobs) {
      port.postMessage({ start: job.index } satisfies FromWorker)
      const r = await runExample(job.code, { loaders: moduleLoaders, timeoutMs: 15_000 })
      const problems: string[] = []
      if (r.error !== undefined)
        problems.push(`${r.error.line === undefined ? '' : `line ${r.error.line + 1}: `}${r.error.message}`)
      for (const e of r.expectations) {
        if (e.status === 'pass' || (e.status === 'skipped' && r.error !== undefined)) continue
        const where = `line ${e.line + 1}: ${e.expression.replace(/\s+/g, ' ').slice(0, 100)}`
        problems.push(
          `${where} // => ${e.kind === 'value' ? e.expected : e.kind}: ${e.message || e.status} (got ${e.actual.slice(0, 200)})`,
        )
      }
      port.postMessage({ result: { index: job.index, ok: r.ok, ms: r.ms, problems } } satisfies FromWorker)
    }
  })
  port.postMessage({ ready: true } satisfies FromWorker)
}

function option(args: readonly string[], name: string): string | undefined {
  const at = args.indexOf(name)
  return at >= 0 ? args[at + 1] : undefined
}

async function main(): Promise<void> {
  const args = process.argv.slice(2)
  const t0 = performance.now()
  const { extractCatalogue } = await import('../packages/meta/src/extract/index.ts')
  const catalogue = extractCatalogue({ root: ROOT, packages: ['engine'] })
  const tExtract = performance.now() - t0
  if (args.includes('--coverage')) {
    const { coverageReport } = await import('../packages/meta/src/extract/index.ts')
    const r = coverageReport(catalogue)
    const counts = Object.entries(r.counts)
      .map(([k, n]) => `${n} ${k}`)
      .join(', ')
    process.stdout.write(`coverage: ${r.complete} of ${r.symbols} core symbols complete; missing: ${counts}\n`)
    for (const i of r.issues.slice(0, Number(option(args, '--coverage-list') ?? 0)))
      process.stdout.write(`  ${i.file}:${i.line}  ${i.symbol}  ${i.problem}${i.detail === '' ? '' : ` ${i.detail}`}\n`)
    const { summaryCoverage } = await import('../packages/meta/src/extract/index.ts')
    const tree = extractCatalogue({ root: ROOT, packages: ['engine', 'render', 'meta', 'hub'] })
    const s = summaryCoverage(tree)
    process.stdout.write(
      `summaries: ${s.folders - s.missingFolders.length} of ${s.folders} folders, ${s.files - s.missingFiles.length} of ${s.files} files\n`,
    )
    for (const path of [...s.missingFolders, ...s.missingFiles]) process.stdout.write(`  no summary: ${path}\n`)
  }
  const only = option(args, '--only')
    ?.split(',')
    .filter((p) => p !== '')
  const withMethods = args.includes('--methods')
  const jobs: Job[] = []
  for (const s of catalogue.symbols) {
    const strict = !s.module.startsWith('whitebox-engine/methods/')
    if (!strict && !withMethods) continue
    const examples = [
      ...s.examples.map((e) => ({ e, symbol: s.id })),
      ...s.members.flatMap((m) => m.examples.map((e) => ({ e, symbol: `${s.id}.${m.name}` }))),
    ]
    for (const { e, symbol } of examples) {
      const file = e.source?.file ?? s.source.file
      if (only !== undefined && !only.some((p) => s.module.startsWith(p) || file.startsWith(p))) continue
      jobs.push({ index: jobs.length, symbol, module: s.module, strict, file, line: e.source?.line ?? 0, code: e.code })
    }
  }
  const nWorkers = Math.max(
    1,
    Math.min(
      Number(option(args, '--workers') ?? process.env['WHITEBOX_EXAMPLE_WORKERS'] ?? 2),
      availableParallelism(),
      jobs.length,
    ),
  )
  const timeout = Number(option(args, '--timeout') ?? 20_000)
  const quiet = args.includes('--quiet')
  process.stdout.write(
    `examples: ${jobs.length} from ${catalogue.symbols.length} symbols (catalogue in ${(tExtract / 1000).toFixed(1)} s), ${nWorkers} workers\n`,
  )

  // Group jobs by module so each worker imports few modules; hand out one module at a time.
  const queue: Job[][] = []
  const byModule = new Map<string, Job[]>()
  for (const j of jobs) byModule.set(j.module, [...(byModule.get(j.module) ?? []), j])
  for (const group of byModule.values()) for (let i = 0; i < group.length; i += 25) queue.push(group.slice(i, i + 25))
  const results = new Map<number, JobResult>()
  let timeouts = 0

  await new Promise<void>((done) => {
    let live = 0
    const spawn = (): void => {
      const batch = queue.shift()
      if (batch === undefined) {
        if (live === 0) done()
        return
      }
      live++
      const w = new Worker(new URL(import.meta.url), { execArgv: [...process.execArgv, ...WORKER_FLAGS] })
      let current: { index: number; since: number } | undefined
      let pending = [...batch]
      let finished = false
      const finish = (): void => {
        if (finished) return
        finished = true
        clearInterval(watchdog)
        live--
        spawn()
      }
      const watchdog = setInterval(() => {
        if (current === undefined || performance.now() - current.since < timeout) return
        const job = jobs[current.index] as Job
        timeouts++
        results.set(job.index, {
          index: job.index,
          ok: false,
          ms: timeout,
          problems: [`timed out after ${timeout} ms`],
        })
        // The rest of the batch goes back on the queue; this worker is replaced.
        const rest = pending.filter((j) => j.index !== job.index && !results.has(j.index))
        if (rest.length > 0) queue.unshift(rest)
        pending = []
        void w.terminate()
        finish()
      }, 250)
      w.on('message', (msg: FromWorker) => {
        if ('ready' in msg) w.postMessage({ jobs: pending } satisfies ToWorker)
        else if ('start' in msg) current = { index: msg.start, since: performance.now() }
        else {
          results.set(msg.result.index, msg.result)
          current = undefined
          if (pending.every((j) => results.has(j.index))) {
            const next = queue.shift()
            if (next === undefined) {
              w.postMessage({ done: true } satisfies ToWorker)
              void w.terminate()
              finish()
            } else {
              pending = [...next]
              w.postMessage({ jobs: next } satisfies ToWorker)
            }
          }
        }
      })
      w.on('error', (e: Error) => {
        for (const j of pending)
          if (!results.has(j.index))
            results.set(j.index, { index: j.index, ok: false, ms: 0, problems: [`worker crashed: ${e.message}`] })
        finish()
      })
      w.on('exit', () => finish())
    }
    for (let i = 0; i < nWorkers; i++) spawn()
  })

  const failed = jobs.filter((j) => results.get(j.index)?.ok !== true)
  const strictFailed = failed.filter((j) => j.strict)
  for (const j of failed) {
    if (quiet && !j.strict) continue
    const r = results.get(j.index)
    const tag = j.strict ? 'FAIL' : 'warn'
    process.stdout.write(`${tag}  ${j.file}:${j.line}  ${j.symbol}\n`)
    for (const p of r?.problems ?? ['no result']) process.stdout.write(`      ${p}\n`)
  }
  const slowest = Number(option(args, '--slowest') ?? 10)
  if (slowest > 0) {
    const top = [...results.values()].sort((a, b) => b.ms - a.ms).slice(0, slowest)
    process.stdout.write(`\nslowest:\n`)
    for (const r of top) {
      const j = jobs[r.index] as Job
      process.stdout.write(`  ${(r.ms / 1000).toFixed(2)} s  ${j.file}:${j.line}  ${j.symbol}\n`)
    }
  }
  const json = option(args, '--json')
  if (json !== undefined) {
    const { writeFileSync } = await import('node:fs')
    writeFileSync(
      json,
      `${JSON.stringify(
        failed.map((j) => ({ ...j, code: undefined, problems: results.get(j.index)?.problems })),
        null,
        1,
      )}\n`,
    )
  }
  const seconds = ((performance.now() - t0) / 1000).toFixed(1)
  process.stdout.write(
    `\nexamples: ${jobs.length} run · ${jobs.length - failed.length} passed · ${strictFailed.length} core failed · ` +
      `${failed.length - strictFailed.length} methods failed · ${timeouts} timed out · ${seconds} s\n`,
  )
  process.exit(strictFailed.length > 0 ? 1 : 0)
}

if (isMainThread) await main()
else await worker()
