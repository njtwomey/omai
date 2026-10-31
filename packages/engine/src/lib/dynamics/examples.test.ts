// Runs every TSDoc @example of the dynamics domain and checks each `// => value` line (`// => throws Code` too).
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { stripTypeScriptTypes } from 'node:module'
import { join, relative } from 'node:path'
import { describe, expect, it } from 'vitest'
import * as autodiff from '../autodiff/index.ts'
import * as base from '../base/index.ts'
import * as numerics from '../numerics/index.ts'
import * as random from '../random/index.ts'
import * as systems from '../systems/index.ts'
import * as tensor from '../tensor/index.ts'
import * as trace from '../trace/index.ts'
import * as dynamics from './index.ts'

const MODULES: Readonly<Record<string, unknown>> = {
  'mdml-engine/autodiff': autodiff,
  'mdml-engine/base': base,
  'mdml-engine/random': random,
  'mdml-engine/systems': systems,
  'mdml-engine/tensor': tensor,
  'mdml-engine/trace': trace,
  'mdml-engine/dynamics': dynamics,
  'mdml-engine/dynamics/celestial': dynamics.celestial,
  'mdml-engine/dynamics/ode': dynamics.ode,
  'mdml-engine/dynamics/pde': (dynamics as Record<string, unknown>)['pde'],
  'mdml-engine/dynamics/control': (dynamics as Record<string, unknown>)['control'],
  'mdml-engine/dynamics/filtering': (dynamics as Record<string, unknown>)['filtering'],
  ...Object.fromEntries(Object.entries(numerics).map(([k, v]) => [`mdml-engine/numerics/${k}`, v])),
}

interface Example {
  readonly file: string
  readonly line: number
  readonly code: string[]
}

function sources(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name)
    if (statSync(path).isDirectory()) sources(path, out)
    else if (name.endsWith('.ts') && !name.includes('.test') && !name.startsWith('_') && name !== 'index.ts')
      out.push(path)
  }
  return out
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
    out.push({ file: relative(import.meta.dirname, path), line: start + 1, code })
  }
  return out
}

function compile(code: readonly string[]): string {
  const body: string[] = []
  for (const line of code) {
    const imp = /^import\s+(type\s+)?\{([^}]*)\}\s+from\s+'([^']+)'/.exec(line)
    if (imp !== null) {
      if (imp[1] !== undefined) continue
      const names = (imp[2] ?? '')
        .split(',')
        .map((s) => s.trim())
        .filter((s) => s.length > 0 && !s.startsWith('type '))
        .map((s) => s.replace(/\s+as\s+/, ': '))
      body.push(`const {${names.join(', ')}} = __mod(${JSON.stringify(imp[3])});`)
      continue
    }
    const throws = /^(.*?)\s*\/\/ => throws (\w[\w-]*)\s*$/.exec(line)
    if (throws !== null) {
      body.push(`__throws(() => (${throws[1]}), ${JSON.stringify(throws[2])}, ${JSON.stringify(line)});`)
      continue
    }
    const check = /^(.*?)\s*\/\/ => (.*)$/.exec(line)
    if (check !== null) {
      const want = check[2] ?? ''
      if (want === 'trace' || want === 'figure') body.push(`(${check[1]});`)
      else body.push(`__check(() => (${check[1]}), () => (${want}), ${JSON.stringify(line)});`)
      continue
    }
    body.push(line)
  }
  return stripTypeScriptTypes(body.join('\n'))
}

describe('dynamics documentation examples', () => {
  for (const file of sources(import.meta.dirname)) {
    for (const ex of examples(file)) {
      it(`${ex.file}:${ex.line}`, () => {
        const run = new Function('__mod', '__check', '__throws', compile(ex.code)) as (
          mod: (name: string) => unknown,
          check: (actual: () => unknown, expected: () => unknown, text: string) => void,
          throws: (f: () => unknown, code: string, text: string) => void,
        ) => void
        run(
          (name) => {
            const m = MODULES[name]
            if (m === undefined) throw new Error(`examples: no module ${name}`)
            return m
          },
          (actual, expected, text) => {
            expect(actual(), text).toEqual(expected())
          },
          (f, code, text) => {
            expect(f, text).toThrow()
            try {
              f()
            } catch (e) {
              expect((e as { code?: string }).code, text).toBe(code)
            }
          },
        )
      })
    }
  }
})
