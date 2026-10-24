import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { extractCatalogue } from './catalogue.ts'

// A miniature repository: a package with a nested namespace, a flat-inlined folder, a re-export and an internal.
const root = mkdtempSync(join(tmpdir(), 'whitebox-meta-'))
const files: Record<string, string> = {
  'packages/mini/package.json': JSON.stringify({
    name: 'mini',
    version: '1.2.3',
    description: 'A miniature package.',
    exports: { './*': { source: './src/lib/*/index.ts' } },
  }),
  'packages/mini/src/lib/core/domain.json': JSON.stringify({ tier: 1, summary: 'The core.', inline: ['ops'] }),
  'packages/mini/src/lib/core/index.ts':
    "export * from './maths.ts'\nexport * from './ops/index.ts'\nexport * as deep from './deep/index.ts'\n",
  'packages/mini/src/lib/core/maths.ts': `/**
 * Add two numbers.
 *
 * @param a - First.
 * @param b - Second.
 * @returns The sum.
 * @example
 * \`\`\`ts
 * import { add } from 'mini/core'
 * add(1, 2) // => 3
 * \`\`\`
 */
export function add(a: number, b = 1): number {
  return a + b
}

/** Internal helper. @internal */
export function secret(): number {
  return 0
}

/** Overloaded. */
export function pick(x: number): number
export function pick(x: string): string
export function pick(x: number | string): number | string {
  return x
}

/** A point. */
export interface Point {
  readonly x: number
}

/** A counter. */
export class Counter {
  /** The count. */
  count = 0
  /**
   * Make one.
   *
   * @param start - Where to start.
   */
  constructor(start: number = 0) {
    this.count = start
  }
  /** Add one. @returns The new count. */
  inc(): number {
    return ++this.count
  }
  private hidden(): void {}
}
`,
  'packages/mini/src/lib/core/ops/index.ts': "export * from './scale.ts'\n",
  'packages/mini/src/lib/core/ops/scale.ts':
    '/** Scale a number.\n *\n * @param x - The number.\n * @param options - Options.\n */\nexport const scale = (x: number, { by = 2 }: { by?: number } = {}): number => x * by\n',
  'packages/mini/src/lib/core/deep/index.ts': "export * from './deep.ts'\n",
  'packages/mini/src/lib/core/deep/deep.ts':
    "import { add } from '../maths.ts'\n/** The answer. */\nexport const answer: number = 42\nexport { add as plus }\n",
}
for (const [path, text] of Object.entries(files)) {
  mkdirSync(dirname(join(root, path)), { recursive: true })
  writeFileSync(join(root, path), text)
}
afterAll(() => rmSync(root, { recursive: true, force: true }))

describe('extractCatalogue', () => {
  const c = extractCatalogue({ root, packages: ['mini'] })
  const sym = (id: string) => c.symbols.find((s) => s.id === id)

  it('finds entry points: namespaces are their own, inlined folders fold into the parent', () => {
    expect(c.modules.map((m) => m.id)).toEqual(['mini/core', 'mini/core/deep'])
    expect(c.modules[0]).toMatchObject({ summary: 'The core.', tier: 1, children: ['mini/core/deep'], layer: 'part' })
    expect(c.packages).toEqual([
      { name: 'mini', description: 'A miniature package.', version: '1.2.3', modules: ['mini/core'] },
    ])
  })

  it('documents each symbol once at its canonical home, without internals', () => {
    expect(c.modules[0]?.symbols).toEqual(['add', 'pick', 'Point', 'Counter', 'scale'])
    expect(sym('mini/core#secret')).toBeUndefined()
    expect(sym('mini/core#add')?.aliases).toEqual(['mini/core/deep#plus'])
    expect(c.modules[1]?.reexports).toEqual({ plus: 'mini/core#add' })
  })

  it('reads signatures, parameters and examples with their source lines', () => {
    const add = sym('mini/core#add')
    expect(add?.signature).toBe('function add(a: number, b = 1): number')
    expect(add?.params.map((p) => [p.name, p.type, p.optional, p.defaultValue, p.description])).toEqual([
      ['a', 'number', false, '', 'First.'],
      ['b', '', true, '1', 'Second.'],
    ])
    expect(add?.source).toEqual({ file: 'packages/mini/src/lib/core/maths.ts', line: 13 })
    expect(add?.examples[0]?.source).toEqual({ file: 'packages/mini/src/lib/core/maths.ts', line: 9 })
    expect(sym('mini/core#pick')?.signature).toBe('function pick(x: number): number\nfunction pick(x: string): string')
    expect(sym('mini/core#scale')?.params.map((p) => p.name)).toEqual(['x', 'options', 'options.by'])
    expect(sym('mini/core#scale')?.kind).toBe('function')
    expect(sym('mini/core/deep#answer')?.signature).toBe('const answer: number')
  })

  it('reads public class members', () => {
    const counter = sym('mini/core#Counter')
    expect(counter?.members.map((m) => [m.kind, m.name, m.signature])).toEqual([
      ['property', 'count', 'count'],
      ['constructor', 'constructor', 'constructor(start: number = 0)'],
      ['method', 'inc', 'inc(): number'],
    ])
    expect(counter?.members[1]?.params[0]?.description).toBe('Where to start.')
  })
})
