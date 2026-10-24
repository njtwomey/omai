// Prolog exceptions inside the resolution engine: a ball term thrown by `throw/1` or by a built-in. The class never
// escapes the engine (uncaught balls become a query result with status `error`).
import { atom, compound, num, type Term, variable } from './terms.ts'

/** A Prolog exception carrying its ball. */
export class PrologThrow extends Error {
  constructor(readonly ball: Term) {
    super('prolog exception')
    this.name = 'PrologThrow'
  }
}

/** `error(Formal, context(Indicator, _))`, as SWI-Prolog builds its error terms. */
export function errorTerm(formal: Term, where: string): Term {
  const slash = where.lastIndexOf('/')
  const pi =
    slash > 0 ? compound('/', [atom(where.slice(0, slash)), num(Number(where.slice(slash + 1)))]) : variable('_')
  return compound('error', [formal, compound('context', [pi, variable('_')])])
}

/** Throw an ISO error term. */
export function fail(formal: Term, where: string): never {
  throw new PrologThrow(errorTerm(formal, where))
}

export const instantiationError = (where: string): never => fail(atom('instantiation_error'), where)
export const typeError = (type: string, culprit: Term, where: string): never =>
  fail(compound('type_error', [atom(type), culprit]), where)
export const domainError = (domain: string, culprit: Term, where: string): never =>
  fail(compound('domain_error', [atom(domain), culprit]), where)
export const evaluationError = (what: string, where: string): never =>
  fail(compound('evaluation_error', [atom(what)]), where)
export const representationError = (what: string, where: string): never =>
  fail(compound('representation_error', [atom(what)]), where)
export const existenceError = (kind: string, culprit: Term, where: string): never =>
  fail(compound('existence_error', [atom(kind), culprit]), where)
export const permissionError = (action: string, type: string, culprit: Term, where: string): never =>
  fail(compound('permission_error', [atom(action), atom(type), culprit]), where)
