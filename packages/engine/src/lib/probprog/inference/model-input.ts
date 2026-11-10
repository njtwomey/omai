/**
 * What a method is made for, and how a model is read: a method takes a bound spec (compiled with `toProgram`) or a program
 * closed over its data, and its capability check reads either analysis (`analyseSpec` of the spec, or `analyse` of the
 * program from one run), so the same method serves a model written in the builder and one written as a closure.
 */

import { analyse, type ProgramAnalysis } from '../analysis.ts'
import { analyseSpec, type SpecAnalysis } from '../model/analysis.ts'
import { type BoundModel, toProgram } from '../model/index.ts'
import type { Model } from '../primitives.ts'

/** What a method is made for: a bound spec (compiled with `toProgram`), or a program closed over its data. */
export type MethodModel = BoundModel | Model

/** What `check` reads: the static analysis of a spec, or the analysis of a program from one run. */
export type ModelAnalysis = SpecAnalysis | ProgramAnalysis

/** The program a method runs. */
export function programOf(model: MethodModel): Model {
  return typeof model === 'function' ? model : toProgram(model)
}

/** The analysis `check` reads for a model: `analyseSpec` of a spec, `analyse` of a program. */
export function analysisOf(model: MethodModel): ModelAnalysis {
  return typeof model === 'function' ? analyse(model, undefined) : analyseSpec(model)
}

/** Whether an analysis is the program's (one run) and not the spec's (static). */
export const isProgramAnalysis = (a: ModelAnalysis): a is ProgramAnalysis => 'prototype' in a
