/**
 * The refusal of the model compiler: the error a mapping throws when a node cannot be put on a factor graph, carrying the
 * kind of issue (`FactorIssueKind`) and the reason in words. The driver turns each one into a `FactorIssue`.
 */

import type { FactorIssueKind } from './types.ts'

export class Refusal extends Error {
  constructor(
    readonly kind: FactorIssueKind,
    reason: string,
  ) {
    super(reason)
  }
}
