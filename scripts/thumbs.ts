/**
 * `make thumbs`: the gallery's thumbnails. Opens every recipe page of the site in headless Chrome, light and dark, and
 * saves its first figure as `apps/site/public/thumbs/<section>/<slug>-<theme>.webp`, which the gallery and the front
 * page show in place of the live figures.
 *
 *   node scripts/thumbs.ts                          every recipe that lacks a thumbnail, both themes
 *   node scripts/thumbs.ts --only axes charts/lines-and-curves     a section, or one recipe (section/slug)
 *   node scripts/thumbs.ts --force --theme dark --url http://localhost:5300     redo them, from a running server
 *
 * Options: --only <section | section/slug> ...; --force (redo thumbnails that exist; by default a recipe whose
 * thumbnails are all there is skipped); --theme light|dark|both (default both); --jobs 4 (tabs working at once);
 * --width 1200 --height 900 (the window); --scale 0.8 (the size of the saved image against the figure on screen);
 * --quality 80 (WebP); --timeout 20000 (ms to wait for a page to settle); --hold 3000 (ms to wait before capturing a
 * figure that is still moving); --url <site> (reuse a running server; otherwise this builds the site into a temporary
 * folder and serves it with `vite preview` on --port, 5391 by default, and stops it at the end).
 *
 * How it works: one headless Chrome driven over the DevTools protocol with Node's WebSocket (no dependency), with
 * several tabs taking recipes from a queue, one theme at a time. A page has settled when its figure exists, every chart
 * has drawn, nothing is `aria-busy` and the figure's box has stopped moving; a figure whose pixels still change after
 * that (a player, a stream) is captured `--hold` ms later. A recipe without a figure on its page is skipped and listed
 * at the end, as is a page that hangs the browser (its tab is replaced). The exit code is 1 if any page failed to
 * settle.
 */
import { type ChildProcess, spawn, spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const ROOT = join(import.meta.dirname, '..')
const SITE = join(ROOT, 'apps/site')
const RECIPES = join(SITE, 'src/gallery/recipes')
const OUT = join(SITE, 'public/thumbs')
const CHROME = process.env['CHROME'] ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const COLOUR_KEY = 'omai-colour-mode'

type Theme = 'light' | 'dark'

const opts = {
  only: [] as string[],
  themes: ['light', 'dark'] as Theme[],
  width: 1200,
  height: 900,
  scale: 0.8,
  quality: 80,
  hold: 3000,
  force: false,
  jobs: 4,
  timeout: 20_000,
  port: 5391,
  url: undefined as string | undefined,
}

const argv = process.argv.slice(2)
for (let i = 0; i < argv.length; i++) {
  const a = argv[i]
  const next = (): string => {
    const v = argv[++i]
    if (v === undefined) throw new Error(`${a} needs a value`)
    return v
  }
  if (a === '--only')
    while (argv[i + 1] !== undefined && !argv[i + 1]?.startsWith('--')) opts.only.push(argv[++i] ?? '')
  else if (a === '--theme') {
    const t = next()
    if (t !== 'light' && t !== 'dark' && t !== 'both') throw new Error('--theme is light, dark or both')
    opts.themes = t === 'both' ? ['light', 'dark'] : [t]
  } else if (a === '--width') opts.width = Number(next())
  else if (a === '--height') opts.height = Number(next())
  else if (a === '--scale') opts.scale = Number(next())
  else if (a === '--hold') opts.hold = Number(next())
  else if (a === '--quality') opts.quality = Number(next())
  else if (a === '--force') opts.force = true
  else if (a === '--jobs') opts.jobs = Math.max(1, Number(next()))
  else if (a === '--timeout') opts.timeout = Number(next())
  else if (a === '--port') opts.port = Number(next())
  else if (a === '--url') opts.url = next().replace(/\/+$/, '')
  else throw new Error(`unknown option ${a}`)
}

interface Recipe {
  readonly section: string
  readonly slug: string
}

/** Every recipe on disk as `{ section, slug }`, the slug being the file name without its order prefix. */
function recipes(): Recipe[] {
  const all = readdirSync(RECIPES, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .flatMap((d) =>
      readdirSync(join(RECIPES, d.name))
        .filter((f) => f.endsWith('.tsx'))
        .sort()
        .map((f) => ({ section: d.name, slug: f.replace(/\.tsx$/, '').replace(/^\d+-/, '') })),
    )
  if (opts.only.length === 0) return all
  return all.filter((r) => opts.only.some((o) => o === r.section || o === `${r.section}/${r.slug}`))
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms))

async function waitFor(url: string, ms: number): Promise<void> {
  const end = Date.now() + ms
  while (Date.now() < end) {
    try {
      if ((await fetch(url)).ok) return
    } catch {
      // not up yet
    }
    await sleep(250)
  }
  throw new Error(`${url} did not come up in ${ms} ms`)
}

/** Open a WebSocket. */
async function connect(url: string): Promise<WebSocket> {
  const socket = new WebSocket(url)
  await new Promise<void>((resolve, reject) => {
    socket.addEventListener('open', () => resolve())
    socket.addEventListener('error', () => reject(new Error('could not connect to Chrome')))
  })
  return socket
}

/** A DevTools connection to one page. */
class Page {
  private id = 0
  private readonly waiting = new Map<number, (r: unknown) => void>()
  private readonly socket: WebSocket
  private readonly targetId: string
  private readonly port: number
  private constructor(socket: WebSocket, port: number, targetId: string) {
    this.socket = socket
    this.port = port
    this.targetId = targetId
    socket.addEventListener('message', (e) => {
      const m = JSON.parse(String(e.data)) as { id?: number; result?: unknown; error?: { message: string } }
      if (m.id === undefined) return
      const done = this.waiting.get(m.id)
      this.waiting.delete(m.id)
      done?.(m.error === undefined ? m.result : { error: m.error.message })
    })
  }
  /** A tab in a window of its own: tabs sharing a window are hidden but one, and a hidden page stops animating. */
  static async open(port: number): Promise<Page> {
    const version = (await (await fetch(`http://127.0.0.1:${port}/json/version`)).json()) as {
      webSocketDebuggerUrl: string
    }
    const browser = await connect(version.webSocketDebuggerUrl)
    const made = await new Promise<string>((resolve, reject) => {
      browser.addEventListener('message', (e) => {
        const m = JSON.parse(String(e.data)) as { result?: { targetId: string }; error?: { message: string } }
        if (m.result !== undefined) resolve(m.result.targetId)
        else reject(new Error(m.error?.message ?? 'could not open a window'))
      })
      browser.send(
        JSON.stringify({ id: 1, method: 'Target.createTarget', params: { url: 'about:blank', newWindow: true } }),
      )
    })
    browser.close()
    return new Page(await connect(`ws://127.0.0.1:${port}/devtools/page/${made}`), port, made)
  }
  /** Send a command; rejects if Chrome has not answered in 30 s (a page stuck in a long script never does). */
  send(method: string, params: object = {}): Promise<Record<string, unknown>> {
    const id = ++this.id
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.waiting.delete(id)
        reject(new Error(`${method} timed out`))
      }, 30_000)
      this.waiting.set(id, (r) => {
        clearTimeout(timer)
        resolve(r as Record<string, unknown>)
      })
      this.socket.send(JSON.stringify({ id, method, params }))
    })
  }
  /** Evaluate an expression in the page and return its value. */
  async eval<T>(expression: string): Promise<T | undefined> {
    const r = (await this.send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })) as {
      result?: { value?: T }
    }
    return r.result?.value
  }
  /** Close the connection and the tab (a tab stuck in a script is dropped with it). */
  close(): void {
    this.socket.close()
    fetch(`http://127.0.0.1:${this.port}/json/close/${this.targetId}`).catch(() => undefined)
  }
}

/**
 * The page's state as the settle check reads it: the figure's box, whether everything on it has drawn, and a small
 * fingerprint of its pixels (canvases shrunk to 16 x 16, svg by its markup) to tell whether it is still moving.
 */
const PROBE = (fallback: boolean): string => `((fallback) => {
  // The recipe's example is the block before the page's previous/next links (its figure, or with \`fallback\`, all of it).
  const example = document.querySelector('main > div > :nth-last-child(2)')?.firstElementChild?.firstElementChild
  const el = document.querySelector('[data-omai-chart], svg[data-omai-scene]') ?? (fallback ? example : null)
  if (!el) return { found: false, example: !!example }
  const box = el.getBoundingClientRect()
  const charts = [...document.querySelectorAll('[data-omai-chart]')]
  const drawn = charts.every((c) => c.querySelector('canvas, svg') !== null && c.getBoundingClientRect().width > 0)
  const busy = document.querySelector('[aria-busy="true"]') !== null
  const error = /Something went wrong|This example failed/.test(document.body.innerText)
  let print = 0
  const mix = (n) => { print = (print * 31 + n) | 0 }
  const small = document.createElement('canvas')
  small.width = small.height = 16
  const g = small.getContext('2d', { willReadFrequently: true })
  for (const c of el.querySelectorAll('canvas')) {
    try {
      g.clearRect(0, 0, 16, 16)
      g.drawImage(c, 0, 0, 16, 16)
      for (const v of g.getImageData(0, 0, 16, 16).data) mix(v)
    } catch { mix(-1) }
  }
  const html = el.outerHTML
  for (let i = 0; i < html.length; i += 7) mix(html.charCodeAt(i))
  return { found: true, example: !!example, drawn, busy, error, print, x: box.x + scrollX, y: box.y + scrollY, w: box.width, h: box.height,
    fonts: document.fonts.status === 'loaded' }
})(${fallback})`

interface Probe {
  found: boolean
  example?: boolean
  drawn?: boolean
  busy?: boolean
  error?: boolean
  print?: number
  fonts?: boolean
  x?: number
  y?: number
  w?: number
  h?: number
}

let vite: ChildProcess | undefined
let chrome: ChildProcess | undefined
let profile: string | undefined
let built: string | undefined
function cleanup(): void {
  chrome?.kill('SIGKILL')
  vite?.kill('SIGTERM')
  for (const dir of [profile, built]) {
    try {
      if (dir !== undefined) rmSync(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 })
    } catch {
      // Chrome may still be writing to its profile as it exits; the system clears the temporary directory
    }
  }
}
process.on('exit', cleanup)
process.on('SIGINT', () => process.exit(130))
process.on('SIGTERM', () => process.exit(143))

/** Build the site into a temporary folder and serve it with `vite preview`; returns its address. */
async function serveBuild(): Promise<string> {
  const node = process.execPath
  const run = (args: string[], what: string): void => {
    const r = spawnSync(node, args, { cwd: SITE, encoding: 'utf8' })
    if (r.status !== 0) throw new Error(`${what} failed:\n${r.stdout}\n${r.stderr}`)
  }
  built = mkdtempSync(join(tmpdir(), 'omai-thumbs-site-'))
  const viteBin = join(SITE, 'node_modules/vite/bin/vite.js')
  console.log('thumbs: building the site')
  run([join(SITE, 'plugins/build-docs.ts'), '--if-stale'], 'the docs catalogue')
  run([viteBin, 'build', '--outDir', built, '--emptyOutDir'], 'the site build')
  vite = spawn(node, [viteBin, 'preview', '--outDir', built, '--port', String(opts.port), '--strictPort'], {
    cwd: SITE,
    stdio: 'ignore',
  })
  const base = `http://localhost:${opts.port}`
  await waitFor(`${base}/`, 30_000)
  return base
}

async function main(): Promise<number> {
  const all = recipes()
  if (all.length === 0) throw new Error(`no recipe matches ${opts.only.join(' ')}`)
  const fileOf = (r: Recipe, theme: Theme): string => join(OUT, r.section, `${r.slug}-${theme}.webp`)
  // By default only what is missing: a recipe whose thumbnails (in the chosen themes) all exist is left alone.
  const todo = opts.force ? all : all.filter((r) => opts.themes.some((t) => !existsSync(fileOf(r, t))))
  if (todo.length === 0) {
    console.log(`thumbs: all ${all.length} recipes have their thumbnails (--force to redo them)`)
    return 0
  }
  const base = opts.url ?? (await serveBuild())
  await waitFor(`${base}/`, 30_000)
  profile = mkdtempSync(join(tmpdir(), 'omai-thumbs-'))
  chrome = spawn(
    CHROME,
    [
      '--headless=new',
      '--remote-debugging-port=0',
      `--user-data-dir=${profile}`,
      '--hide-scrollbars',
      '--no-first-run',
      '--disable-gpu-vsync',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
      '--disable-backgrounding-occluded-windows',
      `--window-size=${opts.width},${opts.height}`,
      'about:blank',
    ],
    { stdio: 'ignore' },
  )
  const portFile = join(profile, 'DevToolsActivePort')
  let port = 0
  for (let i = 0; i < 100 && port === 0; i++) {
    await sleep(200)
    try {
      port = Number(readFileSync(portFile, 'utf8').split('\n')[0])
    } catch {
      // not written yet
    }
  }
  if (port === 0) throw new Error('Chrome did not start')
  const openPage = async (): Promise<Page> => {
    const p = await Page.open(port)
    await p.send('Page.enable')
    await p.send('Emulation.setDeviceMetricsOverride', {
      width: opts.width,
      height: opts.height,
      deviceScaleFactor: 1,
      mobile: false,
    })
    return p
  }

  const failed: string[] = []
  const skipped: string[] = []
  const animated: string[] = []
  let saved = 0

  /** One recipe in the page's current theme. Returns when it is saved, skipped or failed. */
  const shoot = async (page: Page, r: Recipe, theme: Theme): Promise<void> => {
    const name = `${r.section}/${r.slug}`
    await page.send('Page.navigate', { url: `${base}/gallery/${name}` })
    const end = Date.now() + opts.timeout
    let probe: Probe = { found: false }
    let exampleSince = 0
    let fallback = false
    const seen: Probe[] = []
    let stable = false
    while (Date.now() < end) {
      await sleep(100)
      // Pages without a chart or scene (controls, tables, formulas) show their whole example, once it has stood alone a while.
      fallback = exampleSince > 0 && Date.now() - exampleSince > 1500
      probe = (await page.eval<Probe>(PROBE(fallback))) ?? { found: false }
      if (exampleSince === 0 && probe.example === true) exampleSince = Date.now()
      const ready = probe.found && probe.drawn === true && probe.busy !== true && probe.fonts === true
      const box = (p: Probe): string => JSON.stringify([p.x, p.y, p.w, p.h])
      seen.push(probe)
      if (seen.length > 3) seen.shift()
      const first = seen[0]
      if (ready && seen.length === 3 && first !== undefined && seen.every((p) => p.found && box(p) === box(first))) {
        stable = true
        break
      }
      if (!ready) seen.length = 0
    }
    if (!probe.found) {
      skipped.push(name)
      return
    }
    if (!stable || probe.error === true) {
      failed.push(name)
      return
    }
    // The fingerprint moved between the polls: the figure is playing, so wait for it to get somewhere.
    // A player that waits at step 0 is started, then treated the same way.
    const started =
      (await page.eval<boolean>(`(() => {
        const b = document.querySelector('main button[aria-label="Play"]')
        b?.click()
        return b != null
      })()`)) === true
    if (started || seen[0]?.print !== seen[2]?.print) {
      animated.push(name)
      await sleep(opts.hold)
      probe = (await page.eval<Probe>(PROBE(fallback))) ?? probe
    }
    const shot = (await page.send('Page.captureScreenshot', {
      format: 'webp',
      quality: opts.quality,
      captureBeyondViewport: true,
      clip: { x: probe.x, y: probe.y, width: probe.w, height: probe.h, scale: opts.scale },
    })) as { data?: string }
    if (shot.data === undefined) {
      failed.push(name)
      return
    }
    mkdirSync(join(OUT, r.section), { recursive: true })
    writeFileSync(fileOf(r, theme), Buffer.from(shot.data, 'base64'))
    saved++
  }

  const pages = await Promise.all(Array.from({ length: Math.min(opts.jobs, todo.length) }, openPage))
  for (const theme of opts.themes) {
    // The site reads the chosen colour mode before its first paint: write it (the tabs share it), then load every page.
    const first = pages[0]
    if (first === undefined) break
    await first.send('Page.navigate', { url: `${base}/` })
    await sleep(1000)
    await first.eval(`localStorage.setItem('${COLOUR_KEY}', '${theme}')`)
    const queue = opts.force ? [...todo] : todo.filter((r) => !existsSync(fileOf(r, theme)))
    await Promise.all(
      pages.map(async (_, i) => {
        for (let r = queue.shift(); r !== undefined; r = queue.shift()) {
          const page = pages[i]
          if (page === undefined) return
          try {
            await shoot(page, r, theme)
          } catch (e) {
            // A page that never answers (stuck in a script) costs only itself: drop its tab and carry on in a new one.
            failed.push(`${r.section}/${r.slug}`)
            console.log(`thumbs: ${r.section}/${r.slug} (${theme}): ${e instanceof Error ? e.message : String(e)}`)
            page.close()
            pages[i] = await openPage()
          }
        }
      }),
    )
  }
  for (const p of pages) p.close()
  console.log(`thumbs: ${saved} saved for ${todo.length} recipes, ${opts.themes.length} theme(s)`)
  if (skipped.length > 0) console.log(`thumbs: no figure on the page: ${[...new Set(skipped)].join(', ')}`)
  if (animated.length > 0)
    console.log(`thumbs: moving, captured ${opts.hold} ms later: ${[...new Set(animated)].join(', ')}`)
  if (failed.length > 0) console.log(`thumbs: did not settle: ${[...new Set(failed)].join(', ')}`)
  return failed.length > 0 ? 1 : 0
}

main().then(
  (code) => process.exit(code),
  (e: unknown) => {
    console.error(e)
    process.exit(1)
  },
)
