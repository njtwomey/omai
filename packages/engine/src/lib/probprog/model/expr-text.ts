/**
 * Specs and their parts as text: an expression as infix arithmetic and calls, a distribution as `Normal(0, 2)`. Used by
 * the drawing (`toStructure`) and by the structure reports, which quote what they found in the model's own notation.
 */

import type { DistSpec, ExprSpec } from './spec.ts'

const INFIX: Readonly<Record<string, string>> = {
  add: '+',
  sub: '-',
  mul: '*',
  div: '/',
  pow: '^',
  gt: '>',
  lt: '<',
  eq: '=',
}

/**
 * An expression as text: infix arithmetic, function calls, `a[idx]` for a gather, `sum_dim(a)` for a reduction.
 *
 * @param e - The expression.
 * @returns The text.
 * @example
 * ```ts
 * import { exprText } from 'omai-engine/probprog/model'
 * exprText({ kind: 'op', op: 'add', args: [{ kind: 'ref', name: 'mu' }, { kind: 'const', value: 1 }] }) // => 'mu + 1'
 * ```
 */
export function exprText(e: ExprSpec): string {
  const inner = (x: ExprSpec): string => (x.kind === 'op' && x.op in INFIX ? `(${exprText(x)})` : exprText(x))
  switch (e.kind) {
    case 'const':
      return JSON.stringify(e.value)
    case 'ref':
      return e.name
    case 'custom':
      return `${e.name}(${e.args.map(exprText).join(', ')})`
    case 'op': {
      const sym = INFIX[e.op]
      const [a, b] = e.args
      if (sym !== undefined && a !== undefined && b !== undefined) return `${inner(a)} ${sym} ${inner(b)}`
      if (e.op === 'neg' && a !== undefined) return `-${inner(a)}`
      if (e.op === 'matmul' && a !== undefined && b !== undefined) return `${inner(a)} @ ${inner(b)}`
      if (e.op === 'at' && a !== undefined && b !== undefined) return `${inner(a)}[${exprText(b)}]`
      const dim = typeof e.params?.['dim'] === 'string' ? `_${e.params['dim']}` : ''
      return `${e.op}${dim}(${e.args.map(exprText).join(', ')})`
    }
  }
}

/**
 * A distribution as text, `Normal(0, 2)`, its parameters in the family's order.
 *
 * @param d - The distribution spec.
 * @returns The text.
 * @example
 * ```ts
 * import { distText } from 'omai-engine/probprog/model'
 * distText({ family: 'normal', params: { loc: { kind: 'const', value: 0 }, scale: { kind: 'const', value: 2 } } }) // => 'Normal(0, 2)'
 * ```
 */
export function distText(d: DistSpec): string {
  const name = d.family.charAt(0).toUpperCase() + d.family.slice(1)
  return `${name}(${Object.values(d.params).map(exprText).join(', ')})`
}
