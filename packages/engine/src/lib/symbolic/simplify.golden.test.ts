// Rewrites against SymPy. For expand, factor, collect, together, cancel and apart equality is textual: the result
// prints exactly as SymPy's. trigsimp and simplify choose among equivalent forms differently, so for them equality is
// numeric (values at the sample points), and the result must not have more operations than SymPy's.
import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { expectValues, symbolicOutput } from '../../testing/symbolic.ts'
import {
  apart,
  cancel,
  collect,
  countOps,
  type Expr,
  expand,
  factor,
  parse,
  simplify,
  sym,
  together,
  toText,
  trigSimplify,
} from './index.ts'

const fixture = loadFixture('symbolic/simplify')

const ops: Readonly<Record<string, (e: Expr, arg?: string) => Expr>> = {
  expand: (e) => expand(e),
  factor: (e) => factor(e),
  collect: (e, v) => collect(e, parse(v ?? 'x')),
  together: (e) => together(e),
  cancel: (e) => cancel(e),
  apart: (e, v) => apart(e, sym(v ?? 'x')),
  trigsimp: (e) => trigSimplify(e),
  simplify: (e) => simplify(e),
}

describe('simplification (sympy)', () => {
  for (const c of fixture.cases) {
    it(`${c.name}: ${String(c.inputs[0])}`, () => {
      const out = symbolicOutput(c)
      const op = ops[c.fn]
      if (op === undefined) throw new Error(`unknown fn ${c.fn}`)
      const result = op(parse(String(c.inputs[0])), c.inputs[1] === undefined ? undefined : String(c.inputs[1]))
      if (c.params['exact'] === true) expect(toText(result)).toBe(out.text)
      else expect(countOps(result)).toBeLessThanOrEqual(countOps(parse(out.text)))
      expectValues(result, out, c.rtol, c.atol, c.name)
    })
  }
})
