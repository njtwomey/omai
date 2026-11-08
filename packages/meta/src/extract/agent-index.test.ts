import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import type { IndexEntry } from '../schema/agent-index.ts'
import {
  AGENT_INDEX_PACKAGES,
  buildAgentIndex,
  buildRepositoryIndex,
  collectAgentIndex,
  openingSentences,
  renderAgentIndex,
  renderRootIndex,
  staleAgentIndex,
} from './agent-index.ts'
import { extractCatalogue } from './catalogue.ts'

// A tiny repository with the engine's layout: two domains, one with a namespaced and one with a flat subfolder.
const root = mkdtempSync(join(tmpdir(), 'omai-agent-index-'))
const files: Record<string, string> = {
  'packages/engine/package.json': JSON.stringify({
    name: 'omai-engine',
    exports: { './*': { source: './src/lib/*/index.ts' } },
  }),
  'packages/engine/llms.header.md': '# mini-engine\n\nUsage note, written by hand.\n',
  'packages/engine/src/lib/alpha/index.ts':
    "/**\n * Alpha: the first domain. It has more to say. Third sentence.\n */\nexport * from './solve.ts'\nexport * from './ops/index.ts'\nexport * as deep from './deep/index.ts'\n",
  'packages/engine/src/lib/alpha/solve.ts': `/**
 * Solvers, the hub of alpha.
 *
 * @index function Solvers hub: open this file first.
 */

/**
 * Solve it.
 *
 * @param a - Input.
 * @returns Output.
 * @index function Solve a thing.
 * @example
 * \`\`\`ts
 * solve(1) // => 1
 * \`\`\`
 */
export function solve(a: number): number {
  return a
}

/**
 * A fitter.
 *
 * @index estimator Fit a thing.
 */
export const fitter = { name: 'fitter' }

/**
 * An untagged helper.
 */
export function helper(): number {
  return 0
}

/**
 * A second function, tagged after a table so grouping is visible.
 *
 * @index function Another function.
 */
export function another(): number {
  return 1
}

/** @index table Solver choices as one slot. */
export const solvers = {}

/**
 * Hidden.
 *
 * @internal
 * @index function Internal things are not indexed.
 */
export function hidden(): number {
  return 2
}
`,
  'packages/engine/src/lib/alpha/ops/index.ts':
    "/**\n * Operations.\n *\n * @inline\n */\nexport * from './scale.ts'\n",
  'packages/engine/src/lib/alpha/ops/scale.ts': `/**
 * Scaling, flat in alpha.
 */

/**
 * Scale.
 *
 * @index function Scale by a factor.
 */
export function scale(x: number): number {
  return x * 2
}
`,
  'packages/engine/src/lib/alpha/deep/index.ts': "/**\n * Deep things.\n */\nexport * from './deeper.ts'\n",
  'packages/engine/src/lib/alpha/deep/deeper.ts': `/**
 * Deeper things, a hub.
 *
 * @index structure Deep structures hub.
 */

/**
 * Dive.
 *
 * @index function Dive deeper.
 */
export function dive(): number {
  return 3
}
`,
  'packages/engine/src/lib/beta/index.ts': "/**\n * Beta: the second domain.\n */\nexport * from './bare.ts'\n",
  'packages/engine/src/lib/beta/bare.ts': `/**
 * Beta has no tags yet.
 */

/** Bare. */
export function bare(): number {
  return 4
}
`,
}
for (const [path, text] of Object.entries(files)) {
  mkdirSync(dirname(join(root, path)), { recursive: true })
  writeFileSync(join(root, path), text)
}
afterAll(() => rmSync(root, { recursive: true, force: true }))

describe('collectAgentIndex', () => {
  const { entries, summaries, problems, packageName } = collectAgentIndex(root)
  const names = entries.map((e) => `${e.level}:${e.folder}:${e.name}`)

  it('finds file-level and symbol-level entries, each keyed by import path and name', () => {
    expect(packageName).toBe('omai-engine')
    expect(names).toContain('file:alpha:solve.ts')
    expect(names).toContain('symbol:alpha:solve')
    const solve = entries.find((e) => e.name === 'solve')
    expect(solve).toEqual({
      kind: 'function',
      name: 'solve',
      level: 'symbol',
      importPath: 'omai-engine/alpha',
      path: 'src/lib/alpha/solve.ts',
      line: 18,
      description: 'Solve a thing.',
      domain: 'alpha',
      folder: 'alpha',
    })
    const hub = entries.find((e) => e.name === 'solve.ts')
    expect(hub).toMatchObject({
      level: 'file',
      line: 1,
      kind: 'function',
      description: 'Solvers hub: open this file first.',
    })
  })

  it('skips untagged and internal declarations', () => {
    expect(names.some((n) => n.endsWith(':helper') || n.endsWith(':hidden'))).toBe(false)
  })

  it('reads a one-line doc comment that carries a tag', () => {
    expect(entries.find((e) => e.name === 'solvers')).toMatchObject({ kind: 'table', line: 46 })
  })

  it('puts a flat subfolder under its parent entry point and a namespace under its own', () => {
    expect(entries.find((e) => e.name === 'scale')).toMatchObject({
      importPath: 'omai-engine/alpha',
      folder: 'alpha/ops',
    })
    expect(entries.find((e) => e.name === 'dive')).toMatchObject({
      importPath: 'omai-engine/alpha/deep',
      folder: 'alpha/deep',
      domain: 'alpha',
    })
  })

  it('sorts by domain, folder, files first, then kind and name', () => {
    expect(names).toEqual([
      'file:alpha:solve.ts',
      'symbol:alpha:another',
      'symbol:alpha:solve',
      'symbol:alpha:fitter',
      'symbol:alpha:solvers',
      'file:alpha/deep:deeper.ts',
      'symbol:alpha/deep:dive',
      'symbol:alpha/ops:scale',
    ])
    const alpha = entries.filter((e) => e.folder === 'alpha').map((e) => `${e.kind}:${e.name}`)
    expect(alpha).toEqual([
      'function:solve.ts',
      'function:another',
      'function:solve',
      'estimator:fitter',
      'table:solvers',
    ])
  })

  it('reports folder summaries by path relative to lib', () => {
    expect(summaries['alpha']).toContain('Alpha: the first domain.')
    expect(summaries['alpha/deep']).toBe('Deep things.')
  })

  it('has no problems for valid tags', () => {
    expect(problems.filter((p) => p.file.includes('alpha/solve.ts'))).toEqual([])
  })
})

describe('collectAgentIndex problems', () => {
  it('reports a bad kind, an empty description and a duplicate tag with file:line', () => {
    const bad = mkdtempSync(join(tmpdir(), 'omai-agent-index-bad-'))
    const src: Record<string, string> = {
      'packages/engine/package.json': files['packages/engine/package.json'] as string,
      'packages/engine/src/lib/gamma/index.ts': "/**\n * Gamma.\n */\nexport * from './bad.ts'\n",
      'packages/engine/src/lib/gamma/bad.ts': `/**
 * Bad tags.
 */

/**
 * One.
 *
 * @index metric Not a kind.
 */
export function one(): number {
  return 1
}

/**
 * Two.
 *
 * @index function
 */
export function two(): number {
  return 2
}

/**
 * Three.
 *
 * @index function First.
 * @index function Second.
 */
export function three(): number {
  return 3
}
`,
    }
    for (const [path, text] of Object.entries(src)) {
      mkdirSync(dirname(join(bad, path)), { recursive: true })
      writeFileSync(join(bad, path), text)
    }
    try {
      const { problems } = collectAgentIndex(bad)
      const file = 'packages/engine/src/lib/gamma/bad.ts'
      expect(problems.map((p) => [p.severity, p.file, p.line])).toEqual([
        ['error', file, 8],
        ['error', file, 17],
        ['error', file, 27],
      ])
      expect(problems[0]?.message).toContain("unknown @index kind 'metric'")
      expect(problems[1]?.message).toContain('needs a description')
      expect(problems[2]?.message).toContain('duplicate')
    } finally {
      rmSync(bad, { recursive: true, force: true })
    }
  })
})

const entry = (over: Partial<IndexEntry>): IndexEntry => ({
  kind: 'function',
  name: 'f',
  level: 'symbol',
  importPath: 'omai-engine/alpha',
  path: 'src/lib/alpha/f.ts',
  line: 3,
  description: 'Does f.',
  domain: 'alpha',
  folder: 'alpha',
  ...over,
})

describe('renderAgentIndex', () => {
  const entries: IndexEntry[] = [
    entry({ name: 'zeta', kind: 'estimator', line: 10 }),
    entry({ name: 'solve', line: 5 }),
    entry({ name: 'hub.ts', level: 'file', line: 1, kind: 'structure', description: 'The hub.' }),
    entry({ name: 'another', line: 8 }),
    entry({
      name: 'dive',
      folder: 'alpha/deep',
      importPath: 'omai-engine/alpha/deep',
      path: 'src/lib/alpha/deep/d.ts',
    }),
    entry({
      name: 'deepest',
      folder: 'alpha/deep/deeper',
      importPath: 'omai-engine/alpha/deep',
      path: 'src/lib/alpha/deep/deeper/e.ts',
    }),
    entry({ name: 'normal', kind: 'distribution', domain: 'beta', folder: 'beta', importPath: 'omai-engine/beta' }),
  ]
  const summaries = {
    alpha: 'Alpha domain. It has a second sentence that is short.',
    'alpha/deep': 'Deep things.',
    beta: 'Beta domain.',
    gamma: 'Gamma has no entries.',
  }
  const out = renderAgentIndex({
    packageName: 'omai-engine',
    source: 'packages/engine/src/lib',
    entries,
    summaries,
    header: '# omai-engine\n\nNote.\n',
  })

  it('writes llms.txt, one file per domain with entries, and the JSON', () => {
    expect([...out.keys()]).toEqual(['index/alpha.md', 'index/beta.md', 'index/index.json', 'llms.txt'])
  })

  it('opens llms.txt with the hand-written note and lists every domain on one line', () => {
    const llms = out.get('llms.txt') as string
    expect(llms.startsWith('# omai-engine\n\nNote.\n\n## Domains\n')).toBe(true)
    expect(llms).toContain(
      '- [alpha](index/alpha.md): Alpha domain. It has a second sentence that is short. (6 entries)',
    )
    expect(llms).toContain('- [beta](index/beta.md): Beta domain. (1 entry)')
    expect(llms).toContain('- gamma: Gamma has no entries. (no entries yet)')
  })

  it('nests folders as sections and groups entries: files first, then by kind', () => {
    const alpha = out.get('index/alpha.md') as string
    const heads = alpha.split('\n').filter((l) => l.startsWith('#'))
    expect(heads).toEqual(['# alpha', '## alpha/deep', '### alpha/deep/deeper'])
    const lines = alpha.split('\n').filter((l) => l.startsWith('- '))
    expect(lines).toEqual([
      '- structure `hub.ts` (file): The hub. `src/lib/alpha/f.ts:1` `omai-engine/alpha`',
      '- function `another`: Does f. `src/lib/alpha/f.ts:8` `omai-engine/alpha`',
      '- function `solve`: Does f. `src/lib/alpha/f.ts:5` `omai-engine/alpha`',
      '- estimator `zeta`: Does f. `src/lib/alpha/f.ts:10` `omai-engine/alpha`',
      '- function `dive`: Does f. `src/lib/alpha/deep/d.ts:3` `omai-engine/alpha/deep`',
      '- function `deepest`: Does f. `src/lib/alpha/deep/deeper/e.ts:3` `omai-engine/alpha/deep`',
    ])
    expect(alpha).toContain('# alpha\n\nAlpha domain. It has a second sentence that is short.\n')
  })

  it('separates the groups with a blank line', () => {
    const alpha = out.get('index/alpha.md') as string
    expect(alpha).toContain('The hub. `src/lib/alpha/f.ts:1` `omai-engine/alpha`\n\n- function `another`')
    expect(alpha).toContain('`omai-engine/alpha`\n\n- estimator `zeta`')
  })

  it('writes every entry to the JSON with its key fields', () => {
    const json = JSON.parse(out.get('index/index.json') as string) as { entries: IndexEntry[]; kinds: string[] }
    expect(json.kinds).toContain('estimator')
    expect(json.entries).toHaveLength(7)
    expect(json.entries[0]).toEqual(
      entry({ name: 'hub.ts', level: 'file', line: 1, kind: 'structure', description: 'The hub.' }),
    )
    const keys = json.entries.map((e) => `${e.importPath}#${e.name}`)
    expect(new Set(keys).size).toBe(keys.length)
  })

  it('is deterministic: the same output for any input order and on every run', () => {
    const again = renderAgentIndex({
      packageName: 'omai-engine',
      source: 'packages/engine/src/lib',
      entries: [...entries].reverse(),
      summaries,
      header: '# omai-engine\n\nNote.\n',
    })
    expect([...again]).toEqual([...out])
    expect(JSON.stringify([...out])).not.toMatch(/20\d\d-\d\d-\d\d/)
  })

  it('keeps two entries with the same name apart by their import path', () => {
    const both = renderAgentIndex({
      packageName: 'omai-engine',
      source: 'packages/engine/src/lib',
      entries: [
        entry({
          name: 'normal',
          kind: 'distribution',
          domain: 'dists',
          folder: 'dists',
          importPath: 'omai-engine/dists',
        }),
        entry({ name: 'normal', domain: 'random', folder: 'random', importPath: 'omai-engine/random' }),
      ],
      summaries: {},
      header: '',
    })
    expect(both.get('index/dists.md')).toContain('distribution `normal`')
    expect(both.get('index/dists.md')).toContain('`omai-engine/dists`')
    expect(both.get('index/random.md')).toContain('function `normal`')
    expect(both.get('index/random.md')).toContain('`omai-engine/random`')
  })
})

describe('openingSentences', () => {
  it('keeps whole sentences while they fit on a line', () => {
    expect(openingSentences('One. Two.')).toBe('One. Two.')
    expect(openingSentences(`First. ${'Y'.repeat(300)}.`)).toBe('First.')
  })

  it('cuts a long first sentence at a comma outside parentheses and code', () => {
    const long = `Things such as \`a, b\` (one, two), ${'word, '.repeat(60)}end.`
    const out = openingSentences(long)
    expect(out.endsWith(' ...')).toBe(true)
    expect(out.length).toBeLessThan(270)
    expect(out).toContain('`a, b` (one, two)')
  })
})

describe('staleAgentIndex', () => {
  it('lists changed, missing and orphaned files', () => {
    const files = new Map([
      ['llms.txt', 'new'],
      ['index/a.md', 'same'],
      ['index/b.md', 'new'],
    ])
    const existing = new Map([
      ['llms.txt', 'old'],
      ['index/a.md', 'same'],
      ['index/gone.md', 'x'],
    ])
    expect(staleAgentIndex(files, existing)).toEqual(['index/b.md', 'index/gone.md (no longer generated)', 'llms.txt'])
    expect(staleAgentIndex(files, files)).toEqual([])
  })
})

describe('buildAgentIndex', () => {
  it('renders from a repository and reads the usage note from llms.header.md', () => {
    const { collected, files: out } = buildAgentIndex(root)
    expect(collected.entries.length).toBeGreaterThan(5)
    expect((out.get('llms.txt') as string).startsWith('# mini-engine\n\nUsage note, written by hand.\n')).toBe(true)
    expect([...out.keys()]).toContain('index/alpha.md')
    expect(out.has('index/beta.md')).toBe(false)
    expect(out.get('llms.txt')).toContain('- beta: Beta: the second domain. (no entries yet)')
  })
})

describe('the catalogue keeps the tag out of the prose', () => {
  const catalogue = extractCatalogue({ root, packages: ['engine'] })

  it('records the tag on the symbol and strips it from the summary and description', () => {
    const solve = catalogue.symbols.find((s) => s.name === 'solve')
    expect(solve?.index).toEqual({ kind: 'function', description: 'Solve a thing.' })
    expect(solve?.summary).toBe('Solve it.')
    expect(solve?.description).toBe('')
    expect(catalogue.symbols.find((s) => s.name === 'helper')?.index).toBeUndefined()
  })

  it('records the tag of a file and strips it from the file summary', () => {
    const file = catalogue.files.find((f) => f.path.endsWith('alpha/solve.ts'))
    expect(file?.index).toEqual({ kind: 'function', description: 'Solvers hub: open this file first.' })
    expect(file?.summary).toBe('Solvers, the hub of alpha.')
    expect(catalogue.files.find((f) => f.path.endsWith('beta/bare.ts'))?.index).toBeUndefined()
  })
})

// A tiny repository with two indexed packages (the engine under src/lib, render directly under src, with a tsx
// component and a nested react folder) and one that is not indexed (meta).
describe('several packages', () => {
  const repo = mkdtempSync(join(tmpdir(), 'omai-agent-index-multi-'))
  const tree: Record<string, string> = {
    ...files,
    'packages/render/package.json': JSON.stringify({
      name: 'omai-render',
      description: 'Figures.',
      exports: { './*': { source: './src/*/index.ts' } },
    }),
    'packages/render/llms.header.md': '# mini-render\n\nRender note.\n',
    'packages/render/src/index.ts': "/**\n * mini-render.\n */\nexport * from './figure/index.ts'\n",
    'packages/render/src/figure/index.ts': "/**\n * The figure model.\n */\nexport * from './figure.ts'\n",
    'packages/render/src/figure/figure.ts': `/**
 * The figure model, a hub.
 *
 * @index structure Figure model hub.
 */

/**
 * A figure.
 *
 * @index function Make a figure.
 */
export function figure(): number {
  return 1
}
`,
    'packages/render/src/react/index.ts':
      "/**\n * React bindings.\n */\nexport * from './Chart.tsx'\nexport * from './session/index.ts'\n",
    'packages/render/src/react/Chart.tsx': `/**
 * The chart component.
 */

/**
 * A chart.
 *
 * @index function React component: a live chart.
 */
export function Chart(): number {
  return 2
}
`,
    'packages/render/src/react/session/index.ts': "/**\n * Sessions.\n */\nexport * from './useSession.ts'\n",
    'packages/render/src/react/session/useSession.ts': `/**
 * The session hook.
 */

/**
 * A session.
 *
 * @index function React hook: run a method live.
 */
export function useSession(): number {
  return 3
}
`,
    'packages/render/src/theme/index.ts': "/**\n * Theme: tokens.\n */\nexport * from './tokens.ts'\n",
    'packages/render/src/theme/tokens.ts':
      '/**\n * Tokens, untagged.\n */\n\n/** Tokens. */\nexport const tokens = {}\n',
    'packages/meta/package.json': JSON.stringify({ name: 'omai-meta', description: 'Docs tooling.' }),
  }
  for (const [path, text] of Object.entries(tree)) {
    mkdirSync(dirname(join(repo, path)), { recursive: true })
    writeFileSync(join(repo, path), text)
  }
  afterAll(() => rmSync(repo, { recursive: true, force: true }))
  const render = AGENT_INDEX_PACKAGES.find((p) => p.package === 'render')
  if (render === undefined) throw new Error('render is not an indexed package')

  it('indexes render with its source directly under src, and the nested react folders under omai-render/react', () => {
    const { entries, packageName, summaries } = collectAgentIndex(repo, render)
    expect(packageName).toBe('omai-render')
    expect(entries.map((e) => `${e.domain}|${e.folder}|${e.level}|${e.name}|${e.importPath}`)).toEqual([
      'figure|figure|file|figure.ts|omai-render/figure',
      'figure|figure|symbol|figure|omai-render/figure',
      'react|react|symbol|Chart|omai-render/react',
      'react|react/session|symbol|useSession|omai-render/react',
    ])
    expect(entries[2]?.path).toBe('src/react/Chart.tsx')
    expect(summaries['react/session']).toBe('Sessions.')
  })

  it('does not mix packages: the engine collects only its own tags', () => {
    const engine = collectAgentIndex(repo)
    expect(engine.entries.every((e) => e.path.startsWith('src/lib/'))).toBe(true)
    expect(engine.entries.some((e) => e.name === 'figure')).toBe(false)
  })

  it('writes per-package files with that package header, source and nested sections', () => {
    const { files: out } = buildAgentIndex(repo, render)
    expect([...out.keys()]).toEqual(['index/figure.md', 'index/index.json', 'index/react.md', 'llms.txt'])
    const llms = out.get('llms.txt') as string
    expect(llms.startsWith('# mini-render\n\nRender note.\n\n## Domains\n')).toBe(true)
    expect(llms).toContain('- [react](index/react.md): React bindings. (2 entries)')
    expect(llms).toContain('- theme: Theme: tokens. (no entries yet)')
    const react = out.get('index/react.md') as string
    expect(react).toContain('from the @index tags in packages/render/src/react. Do not edit.')
    expect(react.split('\n').filter((l) => l.startsWith('#'))).toEqual(['# react', '## react/session'])
    expect(react).toContain('- function `useSession`: React hook: run a method live. `src/react/session/useSession.ts:')
    expect(JSON.parse(out.get('index/index.json') as string)).toHaveProperty('package', 'omai-render')
  })

  it('handles a package with no entries: the header and every domain from the folder summaries', () => {
    const empty = mkdtempSync(join(tmpdir(), 'omai-agent-index-empty-'))
    try {
      for (const [path, text] of Object.entries(tree).filter(([p]) => p.startsWith('packages/render/'))) {
        mkdirSync(dirname(join(empty, path)), { recursive: true })
        writeFileSync(join(empty, path), text.replace(/ \* @index .*\n/g, ''))
      }
      const built = buildAgentIndex(empty, render)
      expect(built.collected.entries).toEqual([])
      expect([...built.files.keys()]).toEqual(['index/index.json', 'llms.txt'])
      const llms = built.files.get('llms.txt') as string
      expect(llms).toContain('- figure: The figure model. (no entries yet)')
      expect(llms).toContain('- theme: Theme: tokens. (no entries yet)')
      expect(llms).not.toContain('](index/figure.md)')
      expect(built.files.get('index/index.json')).toContain('"entries": []')
    } finally {
      rmSync(empty, { recursive: true, force: true })
    }
  })

  it('maps the packages in a repository-level llms.txt, two lines each, and lists the others in one', () => {
    const built = buildRepositoryIndex(repo)
    expect(built.packages.map((p) => p.package.package)).toEqual(['engine', 'render'])
    const lines = built.root.split('\n')
    expect(lines).toContain('- [omai-render](packages/render/llms.txt): Figures.')
    const at = lines.indexOf('- [omai-render](packages/render/llms.txt): Figures.')
    expect(lines[at + 1]).toBe(
      '  Imports as `omai-render/<domain>`; 4 entries in 3 domains; the index files are in `packages/render/index/`.',
    )
    expect(lines.some((l) => l.startsWith('- [omai-engine](packages/engine/llms.txt)'))).toBe(true)
    expect(lines).toContain('- omai-meta: Docs tooling. Not indexed.')
  })

  it('is deterministic and a pure function of the source', () => {
    const a = buildRepositoryIndex(repo)
    const b = buildRepositoryIndex(repo)
    expect(b.root).toBe(a.root)
    for (const [i, pkg] of a.packages.entries()) expect([...(b.packages[i]?.files ?? [])]).toEqual([...pkg.files])
    expect(a.root).not.toMatch(/20\d\d-\d\d-\d\d/)
  })

  it('finds stale files per package: a changed tag changes only its own package', () => {
    const before = buildRepositoryIndex(repo)
    const onDisk = new Map(before.packages.map((p) => [p.package.package, new Map(p.files)]))
    expect(staleAgentIndex(before.packages[1]?.files ?? new Map(), onDisk.get('render') ?? new Map())).toEqual([])
    const edited = tree['packages/render/src/react/Chart.tsx'] as string
    writeFileSync(join(repo, 'packages/render/src/react/Chart.tsx'), edited.replace('a live chart', 'a chart'))
    try {
      const after = buildRepositoryIndex(repo)
      const stale = (i: number, name: string): string[] =>
        staleAgentIndex(after.packages[i]?.files ?? new Map(), onDisk.get(name) ?? new Map())
      expect(stale(0, 'engine')).toEqual([])
      expect(stale(1, 'render')).toEqual(['index/index.json', 'index/react.md'])
      expect(after.root).toBe(before.root)
    } finally {
      writeFileSync(join(repo, 'packages/render/src/react/Chart.tsx'), edited)
    }
  })

  it('validates tags per package: an unknown kind in render is an error in render only', () => {
    writeFileSync(
      join(repo, 'packages/render/src/theme/tokens.ts'),
      '/**\n * Tokens.\n */\n\n/**\n * Tokens.\n *\n * @index component A kind that is not in the vocabulary.\n */\nexport const tokens = {}\n',
    )
    try {
      const r = collectAgentIndex(repo, render)
      expect(r.problems.map((p) => [p.severity, p.file, p.line])).toEqual([
        ['error', 'packages/render/src/theme/tokens.ts', 8],
      ])
      expect(r.problems[0]?.message).toContain("unknown @index kind 'component'")
      expect(collectAgentIndex(repo).problems.filter((p) => p.file.includes('render'))).toEqual([])
    } finally {
      writeFileSync(
        join(repo, 'packages/render/src/theme/tokens.ts'),
        tree['packages/render/src/theme/tokens.ts'] as string,
      )
    }
  })
})

describe('renderRootIndex', () => {
  it('lists an unindexed package on one line and an empty package as having no entries yet', () => {
    const text = renderRootIndex(
      [
        {
          package: 'x',
          collected: { packageName: 'omai-x', description: 'X.', entries: [], summaries: { a: 'A.' }, problems: [] },
        },
      ],
      [{ name: 'omai-y', description: 'Y.' }],
    )
    expect(text).toContain(
      '- [omai-x](packages/x/llms.txt): X.\n  Imports as `omai-x/<domain>`; no entries yet in 1 domain;',
    )
    expect(text).toContain('- omai-y: Y. Not indexed.')
  })
})
