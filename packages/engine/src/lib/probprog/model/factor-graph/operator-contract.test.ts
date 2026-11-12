// The contract between the compiler and the factor operators of the engine: what each mapping needs exists, every operator is claimed by
// a mapping (or is built by hand only), what the mappings emit over the corpus is what they declare, and the engine accepts each factor.
import { describe, expect, it } from 'vitest'
import {
  defaultOperators as factorOperators,
  type OperatorName,
  operatorNames,
} from '../../../message-passing/index.ts'
import { atomBound, epAtoms } from '../../ep/index.ts'
import { benchmarkModels } from '../../models/index.ts'
import { analyseSpec } from '../analysis.ts'
import type { BoundModel } from '../bind.ts'
import { toFactorGraph } from '../factors.ts'
import { defaultOperators as modelOperators } from '../operators.ts'
import { GraphBuilder, VALUE_ALGEBRA_OPERATORS } from './_builder.ts'
import {
  latentMappings,
  nodeMappings,
  observedElementMappings,
  observedNodeMappings,
  observedRowMappings,
} from './_table.ts'

const tables = {
  node: nodeMappings,
  latent: latentMappings,
  observedNode: observedNodeMappings,
  observedRow: observedRowMappings,
  observedElement: observedElementMappings,
}

/**
 * The operators that only a hand-built graph uses: a test, an atom or the engine's own examples writes them out, and no model idiom maps
 * to them (the compiler reaches the same quantity by another operator, said here).
 */
const HAND_BUILT_ONLY: Readonly<Partial<Record<OperatorName, string>>> = {
  'matmul-constant':
    'y = A w for a whole constant matrix: the compiler writes a matrix product as one dot-constant per coordinate',
  'gaussian-joint':
    'a joint Gaussian density over scalar variables: a hand-built Gaussian Markov field; the compiler packs a joint vector as gaussian-mv',
  'is-between': 'the indicator of an interval: the compiler maps a comparison with greater-than',
  'gaussian-mixture':
    'a Gaussian with normal-gamma components: a hand-built mixture; the compiler maps a mixture as a gate over discrete components',
  'probit-row':
    'one probit row on a weight vector: the compiler uses the plated form (probit-plate) or the probit factor on a predictor',
  'logistic-row':
    'one logistic row on a weight vector: the compiler uses the plated form (logistic-plate) or the logistic factor on a predictor',
  spin: 'a +-1 spin with a field: Ising and Boltzmann models are written as graphs, no model family maps to it',
  'spin-row': 'a +-1 spin read through a row of weights: written as graphs, no model family maps to it',
  'mvn-normal-wishart':
    'an observed vector against a normal-Wishart variable: the compiler maps a normal with a Wishart precision as mvn-wishart',
}

const claimed = (): Set<OperatorName> => {
  const out = new Set<OperatorName>(VALUE_ALGEBRA_OPERATORS)
  for (const table of Object.values(tables)) for (const entry of table) for (const name of entry.requires) out.add(name)
  return out
}

describe('the operators the mappings require', () => {
  it('are all operators of the engine registry', () => {
    const registry = factorOperators()
    for (const [table, entries] of Object.entries(tables))
      for (const entry of entries)
        for (const name of entry.requires) expect(registry[name], `${table}/${entry.name}: ${name}`).toBeDefined()
    for (const name of VALUE_ALGEBRA_OPERATORS) expect(registry[name], name).toBeDefined()
  })

  it('cover every operator of the registry, but for the ones that are built by hand only', () => {
    const have = claimed()
    const unclaimed = operatorNames.filter((name) => !have.has(name))
    expect(unclaimed.sort()).toEqual(Object.keys(HAND_BUILT_ONLY).sort())
    for (const reason of Object.values(HAND_BUILT_ONLY)) expect(reason.length).toBeGreaterThan(20)
  })

  it('name each operator of the model registry that a model operator is implemented by', () => {
    const registry = factorOperators()
    for (const [name, def] of Object.entries(modelOperators()))
      if (def.factor !== undefined) expect(registry[def.factor], `${name} -> ${def.factor}`).toBeDefined()
  })
})

/** The compiled corpus: every atom and benchmark model, as EP compiles it and with its rows expanded. */
function* corpus(): Generator<{ readonly key: string; readonly bound: BoundModel }> {
  for (const a of epAtoms) yield { key: `atom/${a.name}`, bound: atomBound(a.problem(a.defaults)) }
  for (const m of benchmarkModels()) if (m.definition !== undefined) yield { key: `model/${m.name}`, bound: m.bound() }
}

describe('what the mappings emit over the corpus', () => {
  it('is declared by the mapping that emitted it (or is the value algebra), and the engine takes every factor', () => {
    const emitted = new Map<string, Set<string>>()
    const stack: string[] = []
    const restore: (() => void)[] = []
    for (const [table, entries] of Object.entries(tables))
      for (const entry of entries) {
        const original = entry.map as (...args: unknown[]) => unknown
        const key = `${table}/${entry.name}`
        ;(entry as { map: unknown }).map = (...args: unknown[]): unknown => {
          stack.push(key)
          try {
            return original(...args)
          } finally {
            stack.pop()
          }
        }
        restore.push(() => ((entry as { map: unknown }).map = original))
      }
    const addFactor = GraphBuilder.prototype.addFactor
    GraphBuilder.prototype.addFactor = function (this: GraphBuilder, f, variable) {
      const key = stack.at(-1) ?? 'outside'
      if (!emitted.has(key)) emitted.set(key, new Set())
      emitted.get(key)?.add(f.operator)
      return addFactor.call(this, f, variable)
    }
    const rejected: string[] = []
    try {
      for (const { key, bound } of corpus())
        for (const plates of ['vectorise', 'expand'] as const) {
          const c = toFactorGraph(bound, { analysis: analyseSpec(bound), plates, packScalars: true })
          for (const issue of c.issues)
            if (issue.code === 'registry') rejected.push(`${key} ${plates}: ${issue.reason}`)
        }
    } finally {
      GraphBuilder.prototype.addFactor = addFactor
      for (const undo of restore) undo()
    }
    expect(rejected).toEqual([])
    const algebra = new Set<string>(VALUE_ALGEBRA_OPERATORS)
    for (const [key, operators] of emitted) {
      const [table, name] = key.split('/') as [keyof typeof tables, string]
      const declared = new Set<string>(tables[table]?.find((e) => e.name === name)?.requires ?? [])
      const undeclared = [...operators].filter((op) => !declared.has(op) && !algebra.has(op))
      expect(undeclared, key).toEqual([])
    }
    expect(emitted.size).toBeGreaterThan(30)
  })
})
