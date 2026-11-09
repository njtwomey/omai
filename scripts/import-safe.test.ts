import { readdirSync, readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

// Importing a package entry must not touch the browser: a server, a worker, a test and a bundler's pre-render all
// import omai-engine and omai-render without a DOM, and every access to `window` and its kin has to sit inside a
// function that runs later. This test imports every entry point of both packages in Node with the browser's globals
// removed and a trap on each of them that records who reads it, and fails on a read from the packages' own source
// (even a `typeof window` guard at the top of a module: do it inside the function that needs it). A dependency's own
// guard at load (katex, echarts and zod look at `document`, `window`, `navigator`) is the dependency's business.
//
// The entries are derived from `exports` in each package.json (a wildcard such as `./*` -> `./src/lib/*/index.ts`
// stands for every folder with an index), so a new folder is covered without a line here.
//
// The exceptions: none. `omai-render/react` imports react and react-dom, which read no global at load, so it is
// imported like the rest. (A page needs a DOM to *render*, not to import.)

const root = resolve(import.meta.dirname, '..')

/** The globals a browser has and Node lacks (or has differently): none may be read while a module loads. */
const BROWSER = [
  'window',
  'document',
  'navigator',
  'self',
  'location',
  'history',
  'screen',
  'localStorage',
  'sessionStorage',
  'indexedDB',
  'Worker',
  'SharedWorker',
  'requestAnimationFrame',
  'cancelAnimationFrame',
  'requestIdleCallback',
  'ResizeObserver',
  'MutationObserver',
  'IntersectionObserver',
  'HTMLElement',
  'Element',
  'Image',
  'ImageData',
  'OffscreenCanvas',
  'AudioContext',
  'customElements',
  'matchMedia',
  'getComputedStyle',
  'devicePixelRatio',
  'innerWidth',
  'innerHeight',
] as const

interface Package {
  readonly name: string
  readonly dir: string
  /** Entries that must be among those derived: a guard that the wildcard expansion finds nested folders. */
  readonly expected: readonly string[]
}

/** The import paths of a package: its exports with their wildcards expanded over the folders that exist. */
function entries(pkg: Package): { readonly path: string; readonly file: string }[] {
  const json = JSON.parse(readFileSync(join(pkg.dir, 'package.json'), 'utf8')) as {
    exports: Record<string, { source?: string } | string>
  }
  const out: { path: string; file: string }[] = []
  for (const [key, value] of Object.entries(json.exports)) {
    const source = typeof value === 'string' ? undefined : value.source
    if (source === undefined) continue
    const star = source.indexOf('*')
    if (star < 0) {
      out.push({ path: key === '.' ? pkg.name : `${pkg.name}/${key.slice(2)}`, file: join(pkg.dir, source) })
      continue
    }
    const prefix = source.slice(0, star)
    const suffix = source.slice(star + 1)
    for (const file of readdirSync(join(pkg.dir, prefix), { recursive: true, encoding: 'utf8' })) {
      if (!`/${file}`.endsWith(suffix)) continue
      const name = file.slice(0, file.length - suffix.length)
      if (name === '') continue
      out.push({ path: `${pkg.name}/${name}`, file: join(pkg.dir, prefix, file) })
    }
  }
  return out.sort((a, b) => a.path.localeCompare(b.path))
}

const traps = new Map<string, PropertyDescriptor | undefined>()
const reads: { name: string; where: string }[] = []

/** Remove the browser globals and put a trap on each: a read records its caller and answers `undefined`. */
function install(): void {
  for (const name of BROWSER) {
    traps.set(name, Object.getOwnPropertyDescriptor(globalThis, name))
    Object.defineProperty(globalThis, name, {
      configurable: true,
      get() {
        // The frame that read the global (the line after the error's own and this getter's): a dependency's own
        // `typeof document` guard at load (katex, echarts, zod) is the dependency's, not ours.
        const caller = (new Error().stack ?? '').split('\n')[2] ?? ''
        reads.push({ name, where: caller.trim() })
        return undefined
      },
      set() {},
    })
  }
}
function uninstall(): void {
  for (const [name, was] of traps) {
    if (was === undefined) delete (globalThis as Record<string, unknown>)[name]
    else Object.defineProperty(globalThis, name, was)
  }
  traps.clear()
  reads.length = 0
}

beforeEach(install)
afterEach(uninstall)

const packages: readonly Package[] = [
  {
    name: 'omai-engine',
    dir: join(root, 'packages/engine'),
    expected: ['omai-engine/base', 'omai-engine/tensor', 'omai-engine/numerics/linalg'],
  },
  {
    name: 'omai-render',
    dir: join(root, 'packages/render'),
    expected: ['omai-render', 'omai-render/charts', 'omai-render/react', 'omai-render/react/session'],
  },
]

// The reads that came directly from the packages' own source.
const own = (): string[] =>
  reads
    .filter((r) => /packages\/(engine|render)\/src\//.test(r.where) && !/\.test\.tsx?/.test(r.where))
    .map((r) => `${r.name} read at ${r.where}`)

describe('importing an entry point touches no browser global', () => {
  for (const pkg of packages) {
    it(`${pkg.name}: every entry of its exports imports in Node`, async () => {
      const list = entries(pkg)
      expect(list.map((e) => e.path)).toEqual(expect.arrayContaining([...pkg.expected]))
      const failures: string[] = []
      for (const entry of list) {
        try {
          await import(/* @vite-ignore */ entry.file)
        } catch (e) {
          failures.push(`${entry.path}: ${e instanceof Error ? e.message : String(e)}`)
        }
      }
      expect(failures).toEqual([])
      expect(own()).toEqual([])
    }, 240_000)
  }
})

describe('the trap', () => {
  it('records a read of a browser global and answers undefined, and puts the global back afterwards', () => {
    const g = globalThis as Record<string, unknown>
    expect(g['window']).toBeUndefined()
    expect(typeof g['document']).toBe('undefined')
    expect(reads.map((r) => r.name)).toEqual(['window', 'document'])
    uninstall()
    expect('window' in g).toBe(false)
    expect(Object.getOwnPropertyDescriptor(globalThis, 'navigator')?.get).toBeDefined()
    install()
  })
})
