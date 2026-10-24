// Shared helpers of the metrics golden tests. Node only; not published.

import type { MetricResult } from '../lib/base/index.ts'
import type { TensorLike } from '../lib/tensor/index.ts'
import type { GoldenCase } from './golden.ts'
import type { Inputs, Params } from './suite.ts'

/** The arguments of a two-input metric case: targets and predictions (from params when strings), then options. */
export function metricArgs(
  inputs: Inputs,
  params: Params,
): [TensorLike | readonly string[], TensorLike | readonly string[], Record<string, unknown>] {
  const { yTrue, yPred, ...rest } = params as Record<string, unknown>
  const weight = inputs[2]
  return [
    (yTrue as readonly string[] | undefined) ?? inputs[0],
    (yPred as readonly string[] | undefined) ?? inputs[1],
    { ...rest, ...(weight === undefined ? {} : { sampleWeight: weight }) },
  ]
}

/** A metric case runner: call the metric on the case's arguments and compare its estimate. */
export function metricValue(
  metric: (
    a: TensorLike | readonly string[],
    b: TensorLike | readonly string[],
    o: Record<string, unknown>,
  ) => MetricResult,
): (inputs: Inputs, params: Params, c: GoldenCase) => number {
  return (inputs, params) => metric(...metricArgs(inputs, params)).estimate.value
}
