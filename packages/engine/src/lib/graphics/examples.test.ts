// Runs every TSDoc @example of the graphics domain (subfolders included) and checks each `// => value` line.
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { describe, expect, it } from 'vitest'
import * as foundations from '../foundations/index.ts'
import * as random from '../random/index.ts'
import * as scene from '../scene/index.ts'
import * as tensor from '../tensor/index.ts'
import * as trace from '../trace/index.ts'
import * as worlds from '../worlds/index.ts'
import * as graphics from './index.ts'

const MODULES: Readonly<Record<string, Readonly<Record<string, unknown>>>> = {
  'omai-engine/graphics': graphics,
  'omai-engine/worlds': worlds,
  'omai-engine/foundations': foundations,
  'omai-engine/scene': scene,
  'omai-engine/random': random,
  'omai-engine/tensor': tensor,
  'omai-engine/trace': trace,
}

interface Example {
  readonly line: number
  readonly code: string[]
}

function examples(path: string): Example[] {
  const text = readFileSync(path, 'utf8').split('\n')
  const out: Example[] = []
  for (let i = 0; i < text.length; i++) {
    if (!/^\s*\*\s*```ts\s*$/.test(text[i] ?? '')) continue
    const code: string[] = []
    const start = i
    for (i++; i < text.length && !/^\s*\*\s*```\s*$/.test(text[i] ?? ''); i++) {
      code.push((text[i] ?? '').replace(/^\s*\*\s?/, ''))
    }
    out.push({ line: start + 1, code })
  }
  return out
}

function compileExample(code: readonly string[]): string {
  const body: string[] = []
  for (const line of code) {
    const imp = /^import\s+\{([^}]*)\}\s+from\s+'([^']+)'/.exec(line)
    if (imp !== null) {
      body.push(`const {${imp[1]}} = __mods[${JSON.stringify(imp[2])}];`)
      continue
    }
    const check = /^(.*?)\s*\/\/ => (.*)$/.exec(line)
    if (check !== null) {
      if (check[2] === 'trace')
        body.push(`__check(() => (${check[1]}) !== undefined, () => true, ${JSON.stringify(line)});`)
      else body.push(`__check(() => (${check[1]}), () => (${check[2]}), ${JSON.stringify(line)});`)
      continue
    }
    body.push(line)
  }
  return body.join('\n')
}

function sources(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir).sort()) {
    const path = join(dir, name)
    if (statSync(path).isDirectory()) sources(path, out)
    else if (name.endsWith('.ts') && !name.includes('.test') && !name.startsWith('_') && name !== 'index.ts')
      out.push(path)
  }
  return out
}

describe('documentation examples', () => {
  for (const path of sources(import.meta.dirname)) {
    const file = relative(import.meta.dirname, path)
    for (const ex of examples(path)) {
      it(`${file}:${ex.line}`, () => {
        const run = new Function('__mods', '__check', compileExample(ex.code)) as (
          mods: typeof MODULES,
          check: (actual: () => unknown, expected: () => unknown, text: string) => void,
        ) => void
        run(MODULES, (actual, expected, text) => {
          expect(actual(), text).toEqual(expected())
        })
      })
    }
  }
})
