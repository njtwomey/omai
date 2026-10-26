/**
 * The hub's index as a module, `virtual:mdml-hub/index` (`HUB_INDEX`, hub-index-data.ts): every entry's card and page
 * header and the module that defines it, so the hub's index pages import no entry and an entry's page only its own.
 *
 * - The index is read in a child Node process (`--conditions=source`, so the packages resolve to their source),
 *   about 1.5 seconds, once when the server starts or the build loads it.
 * - Dev: an edit under `packages/hub/src` reads it again in the background (debounced), and open pages reload when
 *   it changed; until then the previous index is served.
 */
import { spawn } from 'node:child_process'
import { join, normalize, sep } from 'node:path'
import type { Plugin, ViteDevServer } from 'vite'

const VIRTUAL = 'virtual:mdml-hub/index'
const RESOLVED = `\0${VIRTUAL}`
const ROOT = join(import.meta.dirname, '../../..')
const SCRIPT = join(import.meta.dirname, 'hub-index-data.ts')
const HUB = normalize(join(ROOT, 'packages/hub/src')) + sep
const DEBOUNCE_MS = 400

/** Read the index in a child process; resolves to its JSON. */
function read(): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ['--conditions=source', SCRIPT], {
      cwd: ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let out = ''
    let err = ''
    child.stdout.on('data', (d: Buffer) => {
      out += d.toString()
    })
    child.stderr.on('data', (d: Buffer) => {
      err += d.toString()
    })
    child.on('error', reject)
    child.on('exit', (code) => {
      if (code === 0) resolve(out)
      else reject(new Error(`hub index: reading the entries failed (${code})\n${err}`))
    })
  })
}

/** Serves `virtual:mdml-hub/index`; in dev, an edit to the hub's source reads it again. */
export function hubIndex(): Plugin {
  let json: Promise<string> | null = null
  let server: ViteDevServer | null = null
  let timer: ReturnType<typeof setTimeout> | null = null

  const refresh = (): void => {
    const previous = json
    const next = read()
    void next.then(
      async (text) => {
        json = next
        if (text === (await previous?.catch(() => null))) return
        server?.config.logger.info('hub index: entries changed', { timestamp: true })
        const mod = server?.moduleGraph.getModuleById(RESOLVED)
        if (mod !== undefined && mod !== null) server?.moduleGraph.invalidateModule(mod)
        server?.ws.send({ type: 'full-reload' })
      },
      (e: unknown) => server?.config.logger.error(String(e), { timestamp: true }),
    )
  }

  return {
    name: 'mdml-hub-index',
    configureServer(s) {
      server = s
      json = read()
      s.watcher.add(HUB)
      const changed = (file: string): void => {
        if (!normalize(file).startsWith(HUB)) return
        if (timer !== null) clearTimeout(timer)
        timer = setTimeout(() => {
          timer = null
          refresh()
        }, DEBOUNCE_MS)
      }
      s.watcher.on('change', changed)
      s.watcher.on('add', changed)
      s.watcher.on('unlink', changed)
    },
    resolveId(id) {
      return id === VIRTUAL ? RESOLVED : null
    },
    async load(id) {
      if (id !== RESOLVED) return null
      json ??= read()
      return `export const HUB_INDEX = ${await json}\n`
    },
  }
}
