// Runs every TSDoc @example of the logic domain (and its ilp folder) and checks each `// => value` line.
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { describe, expect, it } from 'vitest'
import * as random from '../random/index.ts'
import * as trace from '../trace/index.ts'
import * as ilp from './ilp/index.ts'
import * as logic from './index.ts'

const MODULES: Readonly<Record<string, Readonly<Record<string, unknown>>>> = {
  'omai-engine/logic': logic,
  'omai-engine/logic/ilp': ilp,
  'omai-engine/random': random,
  'omai-engine/trace': trace,
}

interface Example {
  readonly file: string
  readonly line: number
  readonly code: string[]
}

function examples(file: string): Example[] {
  const text = readFileSync(file, 'utf8').split('\n')
  const out: Example[] = []
  for (let i = 0; i < text.length; i++) {
    if (!/^\s*\*\s*```ts\s*$/.test(text[i] ?? '')) continue
    const code: string[] = []
    const start = i
    for (i++; i < text.length && !/^\s*\*\s*```\s*$/.test(text[i] ?? ''); i++) {
      code.push((text[i] ?? '').replace(/^\s*\*\s?/, ''))
    }
    out.push({ file, line: start + 1, code })
  }
  return out
}

function compileExample(code: readonly string[]): string {
  const body: string[] = []
  for (const line of code) {
    const imp = /^import\s+\{([^}]*)\}\s+from\s+'([^']+)'/.exec(line)
    if (imp !== null) {
      body.push(`const {${(imp[1] ?? '').replace(/\btype\s+\w+\s*,?/g, '')}} = __mods[${JSON.stringify(imp[2])}];`)
      continue
    }
    const check = /^(.*?)\s*\/\/ => (.*)$/.exec(line)
    if (check !== null) {
      body.push(`__check(() => (${check[1]}), () => (${check[2]}), ${JSON.stringify(line)});`)
      continue
    }
    body.push(line)
  }
  return body.join('\n')
}

function sources(dir: string): string[] {
  const out: string[] = []
  for (const name of readdirSync(dir)) {
    const path = join(dir, name)
    if (statSync(path).isDirectory()) out.push(...sources(path))
    else if (name.endsWith('.ts') && !name.includes('.test') && !name.startsWith('_') && name !== 'index.ts')
      out.push(path)
  }
  return out
}

describe('documentation examples', () => {
  for (const file of sources(import.meta.dirname)) {
    for (const ex of examples(file)) {
      it(`${relative(import.meta.dirname, file)}:${ex.line}`, () => {
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
