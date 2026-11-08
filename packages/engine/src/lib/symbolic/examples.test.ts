// Runs every TSDoc @example of the symbolic domain and checks each `// => value` line.
import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { type ErrorCode, isOmaiError } from '../base/index.ts'
import * as tensor from '../tensor/index.ts'
import * as symbolic from './index.ts'

const MODULES: Readonly<Record<string, Readonly<Record<string, unknown>>>> = {
  'omai-engine/symbolic': symbolic,
  'omai-engine/tensor': tensor,
}

interface Example {
  readonly file: string
  readonly line: number
  readonly code: string[]
}

function examples(file: string): Example[] {
  const text = readFileSync(join(import.meta.dirname, file), 'utf8').split('\n')
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
      body.push(`const {${imp[1]}} = __mods[${JSON.stringify(imp[2])}];`)
      continue
    }
    const throws = /^(.*?)\s*\/\/ => throws (\w[\w-]*)$/.exec(line)
    if (throws !== null) {
      body.push(`__throws(() => (${throws[1]}), ${JSON.stringify(throws[2])}, ${JSON.stringify(line)});`)
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

const files = readdirSync(import.meta.dirname).filter(
  (f) => f.endsWith('.ts') && !f.includes('.test') && !f.startsWith('_') && f !== 'index.ts',
)

describe('documentation examples', () => {
  for (const file of files) {
    for (const ex of examples(file)) {
      it(`${file}:${ex.line}`, () => {
        const run = new Function('__mods', '__check', '__throws', compileExample(ex.code)) as (
          mods: typeof MODULES,
          check: (actual: () => unknown, expected: () => unknown, text: string) => void,
          throws: (f: () => unknown, code: string, text: string) => void,
        ) => void
        run(
          MODULES,
          (actual, expected, text) => {
            expect(actual(), text).toEqual(expected())
          },
          (f, code, text) => {
            expect(f, text).toThrow()
            try {
              f()
            } catch (e) {
              expect(isOmaiError(e, code as ErrorCode), text).toBe(true)
            }
          },
        )
      })
    }
  }
})
