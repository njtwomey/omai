import { describe, expect, it } from 'vitest'
import { isMdmlError } from '../../base/index.ts'
import { forwardPlan, ground, parseDomain, parseProblem, parseSExpr, pddl } from '../index.ts'

const logistics = `
(define (domain tiny-logistics)
  (:requirements :strips :typing :equality)
  (:types place - object truck package - vehicle-or-thing vehicle-or-thing)
  (:constants depot - place)
  (:predicates (at ?x ?p) (in ?p ?t))
  (:action drive :parameters (?t - truck ?from ?to - place)
    :precondition (and (at ?t ?from) (not (= ?from ?to)))
    :effect (and (at ?t ?to) (not (at ?t ?from))))
  (:action load :parameters (?p - package ?t - truck ?l - place)
    :precondition (and (at ?p ?l) (at ?t ?l))
    :effect (and (in ?p ?t) (not (at ?p ?l))))
  (:action unload :parameters (?p - package ?t - truck ?l - place)
    :precondition (and (in ?p ?t) (at ?t ?l))
    :effect (and (at ?p ?l) (not (in ?p ?t)))))`

const problem = `
(define (problem deliver) (:domain tiny-logistics)
  (:objects shop - place van - truck parcel - package)
  (:init (at van depot) (at parcel depot))
  (:goal (and (at parcel shop))))`

describe('PDDL', () => {
  it('parses s-expressions, typed lists, constants and equality, and grounds to reachable actions', () => {
    expect(parseSExpr('(a (b c) d) (e)')).toEqual([['a', ['b', 'c'], 'd'], ['e']])
    const d = parseDomain(logistics)
    expect(d.types.get('truck')).toBe('vehicle-or-thing')
    expect(d.constants).toEqual([{ name: 'depot', type: 'place' }])
    const p = pddl(logistics, problem)
    expect(p.initial).toEqual(['(at parcel depot)', '(at van depot)'])
    expect(p.actions.some((a) => a.name === '(drive van depot depot)')).toBe(false)
    expect(p.actions.every((a) => !a.name.startsWith('(load parcel van shop') || a.pre.length === 2)).toBe(true)
    const r = forwardPlan(p, { search: 'breadth-first' })
    expect(r.plan).toEqual(['(load parcel van depot)', '(drive van depot shop)', '(unload parcel van shop)'])
  })

  it('rejects input outside the STRIPS subset and malformed input', () => {
    const code = (f: () => unknown): string | undefined => {
      try {
        f()
      } catch (e) {
        return isMdmlError(e) ? e.code : 'other'
      }
      return undefined
    }
    expect(code(() => parseSExpr('(a (b)'))).toBe('value')
    expect(code(() => parseSExpr('a)'))).toBe('value')
    expect(code(() => parseDomain('(define (domain d) (:requirements :adl))'))).toBe('not-implemented')
    expect(
      code(() => parseDomain('(define (domain d) (:action a :parameters () :precondition (not (p)) :effect (q)))')),
    ).toBe('not-implemented')
    expect(
      code(() => parseDomain('(define (domain d) (:action a :parameters () :precondition (or (p) (q)) :effect (q)))')),
    ).toBe('not-implemented')
    expect(code(() => parseDomain('(domain d)'))).toBe('value')
    expect(code(() => parseProblem('(define (problem p) (:domain d) (:goal (not (p))))'))).toBe('not-implemented')
    const d = parseDomain(
      '(define (domain d) (:action a :parameters (?x - thing) :precondition (p ?x) :effect (q ?x)))',
    )
    expect(code(() => ground(d, parseProblem('(define (problem p) (:domain other) (:goal (q)))')))).toBe('value')
    expect(
      code(() => ground(d, parseProblem('(define (problem p) (:domain d) (:objects o - thing) (:goal (q o)))'))),
    ).toBe('value')
  })
})
