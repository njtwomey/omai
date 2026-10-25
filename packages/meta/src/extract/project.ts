/**
 * The packages and their public entry points, read from each `package.json`'s `exports` and the generated barrels: a
 * folder matched by a wildcard export is an entry point when it sits directly under the wildcard's base or its
 * parent's barrel exposes it as a namespace (`export * as linalg`); a folder its parent re-exports flat
 * (`export * from './ops/index.ts'`) belongs to the parent.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import type { ModuleLayer } from '../schema/catalogue.ts'

/** A package to document. */
export interface PackageInfo {
  readonly name: string
  /** Absolute path of the package folder. */
  readonly dir: string
  readonly description: string
  readonly version: string
}

/** One public entry point and the source it owns. */
export interface EntryPoint {
  /** The import specifier, such as `mdml-engine/numerics/linalg`. */
  readonly specifier: string
  readonly package: string
  /** The path below the package (`''` for the root). */
  readonly path: string
  readonly layer: ModuleLayer
  /** Absolute path of its folder. */
  readonly dir: string
  /** Absolute path of its `index.ts`. */
  readonly index: string
  /** Absolute paths of the module files it owns: its own and those of the folders folded into it. */
  readonly files: readonly string[]
  /** Absolute paths of its folder and the folders folded into it (each has an `index.ts` with a summary). */
  readonly folders: readonly string[]
  readonly parent?: string
}

/** The packages and entry points of a repository. */
export interface Project {
  /** Absolute path of the repository root. */
  readonly root: string
  readonly packages: readonly PackageInfo[]
  readonly entries: readonly EntryPoint[]
}

interface PackageJson {
  readonly name: string
  readonly description?: string
  readonly version?: string
  readonly exports?: Readonly<Record<string, unknown>>
}

/**
 * Whether a file is a module of its folder (what the barrels re-export): `.ts` or `.tsx`, not an index, test, type
 * test, declaration or `*.data.ts` payload, and not private (`_name`).
 *
 * @param name - A file name.
 * @returns `true` for module files.
 */
export function isModuleFile(name: string): boolean {
  return (
    /\.tsx?$/.test(name) &&
    !name.startsWith('_') &&
    !/^index\.tsx?$/.test(name) &&
    !/\.test(-d)?\.tsx?$/.test(name) &&
    !name.endsWith('.data.ts') &&
    !name.endsWith('.d.ts')
  )
}

function subdirs(path: string): string[] {
  return readdirSync(path)
    .filter((n) => !n.startsWith('.') && !n.startsWith('_') && n !== 'node_modules')
    .filter((n) => statSync(join(path, n)).isDirectory())
    .sort()
}

function moduleFiles(path: string): string[] {
  return readdirSync(path)
    .filter(isModuleFile)
    .sort()
    .map((n) => join(path, n))
}

function sourceOf(target: unknown): string | undefined {
  if (typeof target === 'string') return target
  if (typeof target === 'object' && target !== null) {
    const t = target as Record<string, unknown>
    const s = t['source'] ?? t['import'] ?? t['default']
    return typeof s === 'string' ? s : undefined
  }
  return undefined
}

/** How a barrel exposes each subfolder: `'namespace'`, `'flat'`, or absent. */
function exposure(index: string): Map<string, 'namespace' | 'flat'> {
  const out = new Map<string, 'namespace' | 'flat'>()
  if (!existsSync(index)) return out
  const text = readFileSync(index, 'utf8')
  for (const m of text.matchAll(/export \* as \w+ from '\.\/([^/']+)\/index\.tsx?'/g))
    out.set(m[1] as string, 'namespace')
  for (const m of text.matchAll(/export \* from '\.\/([^/']+)\/index\.tsx?'/g)) out.set(m[1] as string, 'flat')
  return out
}

function layerOf(pkg: string, path: string): ModuleLayer {
  if (pkg !== 'mdml-engine') return 'part'
  return path.startsWith('methods/') ? 'methods' : 'core'
}

/**
 * Read the given packages of a repository into entry points.
 *
 * @param root - The repository root.
 * @param packages - Package folder names under `packages/`, such as `['engine']`.
 * @returns The packages and their entry points, sorted by specifier.
 */
export function readProject(root: string, packages: readonly string[]): Project {
  const infos: PackageInfo[] = []
  const entries: EntryPoint[] = []
  for (const folder of packages) {
    const dir = resolve(root, 'packages', folder)
    const pkg = JSON.parse(readFileSync(join(dir, 'package.json'), 'utf8')) as PackageJson
    infos.push({ name: pkg.name, dir, description: pkg.description ?? '', version: pkg.version ?? '0.0.0' })
    const exportsMap = pkg.exports ?? {}
    const add = (specifier: string, path: string, entryDir: string, parent?: string): void => {
      const files: string[] = []
      const folders: string[] = []
      const fold = (d: string): void => {
        folders.push(d)
        files.push(...moduleFiles(d))
        const shown = exposure(join(d, 'index.ts'))
        for (const s of subdirs(d)) if (shown.get(s) === 'flat') fold(join(d, s))
      }
      fold(entryDir)
      entries.push({
        specifier,
        package: pkg.name,
        path,
        layer: layerOf(pkg.name, path),
        dir: entryDir,
        index: join(entryDir, 'index.ts'),
        files,
        folders,
        ...(parent === undefined ? {} : { parent }),
      })
      // Namespaced subfolders are entry points of their own; flat ones were folded in above (recursively).
      const walk = (d: string, p: string, spec: string): void => {
        const shown = exposure(join(d, 'index.ts'))
        for (const s of subdirs(d)) {
          if (!existsSync(join(d, s, 'index.ts'))) continue
          if (shown.get(s) === 'namespace') add(`${spec}/${s}`, `${p}/${s}`, join(d, s), spec)
          else if (shown.get(s) === 'flat') walk(join(d, s), `${p}/${s}`, spec)
        }
      }
      walk(entryDir, path, specifier)
    }
    for (const [key, target] of Object.entries(exportsMap)) {
      const source = sourceOf(target)
      if (source === undefined || !/\.tsx?$/.test(source)) continue
      if (!key.includes('*')) {
        const index = resolve(dir, source)
        const path = key === '.' ? '' : key.replace(/^\.\//, '')
        const specifier = path === '' ? pkg.name : `${pkg.name}/${path}`
        if (existsSync(index)) add(specifier, path, dirname(index))
        continue
      }
      // `./methods/*` -> `./src/methods/*/index.ts`: each folder directly under the base is an entry point.
      const prefix = key.replace(/^\.\//, '').replace(/\*$/, '')
      const base = resolve(dir, source.slice(0, source.indexOf('*')))
      if (!existsSync(base)) continue
      for (const s of subdirs(base)) {
        if (!existsSync(join(base, s, 'index.ts'))) continue
        const path = `${prefix}${s}`
        // A plain `./*` wildcard must not also claim folders another wildcard of the same package names.
        if (prefix === '' && Object.keys(exportsMap).some((k) => k !== key && k.startsWith(`./${s}/`))) continue
        add(`${pkg.name}/${path}`, path, join(base, s))
      }
    }
  }
  // A package root (`mdml-render`) is the parent of the package's top-level parts.
  const roots = new Set(entries.filter((e) => e.path === '').map((e) => e.package))
  const linked = entries.map(
    (e): EntryPoint =>
      e.parent === undefined && e.path !== '' && !e.path.includes('/') && roots.has(e.package)
        ? { ...e, parent: e.package }
        : e,
  )
  linked.sort((a, b) => a.specifier.localeCompare(b.specifier))
  return { root: resolve(root), packages: infos, entries: linked }
}

/**
 * The repository-relative path of a file.
 *
 * @param project - The project.
 * @param file - An absolute path.
 * @returns The path relative to the repository root.
 */
export function relativeTo(project: Project, file: string): string {
  return relative(project.root, file)
}
