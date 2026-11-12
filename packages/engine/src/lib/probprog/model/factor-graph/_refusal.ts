/**
 * The refusal of the model compiler: the error a mapping throws when a node cannot be put on a factor graph. It carries a code (what the
 * model has to change: `RefusalCode`, a small closed set), the facts the sentence names (the node, the family, a value) and the sentence
 * (the reason, from the catalogue `_reasons.ts`). The driver turns each one into a `FactorIssue`. `kind` is the coarser class of
 * `FactorIssueKind` that the code belongs to, kept for the readers of an issue.
 */

import type { FactorIssueKind, RefusalCode } from './types.ts'

/** The facts a refusal's sentence names, by the name the catalogue gives them. */
export type RefusalFacts = Readonly<Record<string, string | number | readonly string[]>>

/** The `FactorIssueKind` of each code. */
export const CODE_KIND: Readonly<Record<RefusalCode, FactorIssueKind>> = {
  'no-rule': 'operator',
  'random-parameter': 'operator',
  'wrong-form': 'operator',
  registry: 'operator',
  'random-value': 'node',
  node: 'node',
  missing: 'structure',
  shape: 'structure',
  structure: 'structure',
  support: 'data',
  unbound: 'data',
  constraint: 'constraint',
  family: 'family',
  opaque: 'opaque',
}

export class Refusal extends Error {
  constructor(
    readonly code: RefusalCode,
    reason: string,
    readonly facts: RefusalFacts = {},
  ) {
    super(reason)
  }

  /** The coarse class of the refusal (`FactorIssueKind`), from its code. */
  get kind(): FactorIssueKind {
    return CODE_KIND[this.code]
  }

  /** The node the refusal names, when the sentence names one. */
  get node(): string | undefined {
    const node = this.facts['node']
    return typeof node === 'string' ? node : undefined
  }
}
