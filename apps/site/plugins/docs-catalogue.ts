/**
 * The engine documentation for the docs pages, from the precompiled files of `make docs-catalogue` (docs-data.ts),
 * so no request waits for an extraction.
 *
 * - `virtual:omai-docs/index`: the URL of the current index (`<base>docs-data/index.<hash>.json`); the pages fetch
 *   the index, then the search index and each file's data as they need them.
 * - Dev: the server serves `<base>docs-data/*` from `apps/site/.generated/docs/`. When the files are stale at start, or
 *   after an edit to the engine's source, it regenerates them in a background process (debounced, at low priority)
 *   while still serving the previous version, then tells open pages (`omai-docs:update`) to fetch the new index.
 *   Only when there are no files at all does the first request wait for them.
 * - Build: the files are copied into the bundle as assets under `docs-data/` (regenerated first if stale).
 */
import { type ChildProcess, spawn } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { setPriority } from 'node:os'
import { join, normalize, sep } from 'node:path'
import type { Plugin, ViteDevServer } from 'vite'
import type { WireManifest } from '../src/docs/wire.ts'
import { DOCS_DATA } from '../src/docs/wire.ts'
import { isStale, OUT, ROOT, readManifest, SOURCES } from './docs-data.ts'

const INDEX = 'virtual:omai-docs/index'
const BUILD = join(import.meta.dirname, 'build-docs.ts')
const DEBOUNCE_MS = 400

/** Run build-docs.ts in a child process at low priority; resolves when it exits. */
function generate(log: (msg: string) => void): Promise<void> {
  return new Promise((resolve) => {
    const child: ChildProcess = spawn(process.execPath, [BUILD, '--quiet'], {
      cwd: ROOT,
      stdio: ['ignore', 'ignore', 'pipe'],
    })
    if (child.pid !== undefined)
      try {
        setPriority(child.pid, 10)
      } catch {
        // Not permitted here; run at normal priority.
      }
    let err = ''
    child.stderr?.on('data', (d: Buffer) => {
      err += d.toString()
    })
    child.on('exit', (code) => {
      if (code !== 0) log(`docs catalogue: regeneration failed (${code})\n${err}`)
      resolve()
    })
  })
}

/** Serves the precompiled docs catalogue; an edit to the engine's source regenerates it in the background. */
export function docsCatalogue(): Plugin {
  let base = '/'
  let build = false
  let manifest: WireManifest | null = null
  let server: ViteDevServer | null = null
  // The running regeneration, whether another is wanted after it, and the debounce timer.
  let running: Promise<void> | null = null
  let again = false
  let timer: ReturnType<typeof setTimeout> | null = null

  const log = (msg: string): void => server?.config.logger.info(msg, { timestamp: true })
  const regenerate = (): Promise<void> => {
    if (running !== null) {
      again = true
      return running
    }
    const t0 = performance.now()
    running = generate(log).then(() => {
      running = null
      const next = readManifest()
      if (next !== null && next.version !== manifest?.version) {
        manifest = next
        log(`docs catalogue: regenerated in ${((performance.now() - t0) / 1000).toFixed(1)} s`)
        const mod = server?.moduleGraph.getModuleById(`\0${INDEX}`)
        if (mod !== undefined && mod !== null) server?.moduleGraph.invalidateModule(mod)
        server?.ws.send({ type: 'custom', event: 'omai-docs:update', data: { index: indexUrl() } })
      }
      if (again) {
        again = false
        void regenerate()
      }
    })
    return running
  }
  const indexUrl = (): string => (manifest === null ? '' : `${base}${DOCS_DATA}${manifest.index}`)

  return {
    name: 'omai-docs-catalogue',
    configResolved(config) {
      base = config.base
      build = config.command === 'build'
    },
    async buildStart() {
      // The dev server reads the files in configureServer; `make site` has already brought them up to date.
      if (!build) return
      if (isStale()) await generate((msg) => this.warn(msg))
      manifest = readManifest()
      if (manifest === null) this.error('docs catalogue: no files were generated (run make docs-catalogue)')
    },
    configureServer(s) {
      server = s
      manifest = readManifest()
      if (manifest === null || isStale()) void regenerate()
      const engine = SOURCES.filter((p) => !p.endsWith('.ts') && !p.endsWith('.json'))
      s.watcher.add(engine)
      s.watcher.on('all', (_event, file) => {
        if (!engine.some((dir) => file.startsWith(dir + sep)) || !/\.(tsx?|json)$/.test(file)) return
        if (timer !== null) clearTimeout(timer)
        timer = setTimeout(() => {
          timer = null
          void regenerate()
        }, DEBOUNCE_MS)
      })
      const prefix = `${base}${DOCS_DATA}`
      s.middlewares.use((req, res, next) => {
        const url = req.url?.split('?')[0] ?? ''
        if (!url.startsWith(prefix)) return next()
        const path = normalize(join(OUT, url.slice(prefix.length)))
        if (!path.startsWith(OUT + sep) || !path.endsWith('.json')) return next()
        let body: Buffer
        try {
          body = readFileSync(path)
        } catch {
          res.statusCode = 404
          res.end()
          return
        }
        res.setHeader('Content-Type', 'application/json')
        // Named by content: never changes.
        res.setHeader('Cache-Control', 'public, max-age=31536000, immutable')
        res.end(body)
      })
    },
    resolveId: (id) => (id === INDEX ? `\0${INDEX}` : null),
    async load(id) {
      if (id !== `\0${INDEX}`) return null
      // Only when nothing was ever generated: wait for the first files.
      if (manifest === null && running !== null) await running
      return `export default ${JSON.stringify(indexUrl())}`
    },
    generateBundle() {
      if (manifest === null) return
      for (const name of manifest.files)
        this.emitFile({ type: 'asset', fileName: `${DOCS_DATA}${name}`, source: readFileSync(join(OUT, name)) })
    },
  }
}
