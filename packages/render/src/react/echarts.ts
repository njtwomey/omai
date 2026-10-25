/**
 * The ECharts core the React bindings draw with: `loadEcharts` hands out the shared tree-shaken core from
 * `mdml-render/charts`, with every chart type, both renderers and the mdml themes.
 */

import type * as echarts from 'echarts/core'
import { echartsCore } from '../charts/echarts-core.ts'

/**
 * The ECharts core the React bindings draw with: `echartsCore` from `mdml-render/charts` (every chart type and
 * component the builders use, both renderers, the mdml themes). Idempotent.
 *
 * @returns The ECharts core namespace.
 */
export function loadEcharts(): typeof echarts {
  return echartsCore()
}
