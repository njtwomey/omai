/**
 * The ECharts core the React bindings draw with: `loadEcharts` hands out the shared tree-shaken core from
 * `whitebox-render/charts`, with every chart type, both renderers and the whitebox themes.
 */

import type * as echarts from 'echarts/core'
import { echartsCore } from '../charts/echarts-core.ts'

/**
 * The ECharts core the React bindings draw with: `echartsCore` from `whitebox-render/charts` (every chart type and
 * component the builders use, both renderers, the whitebox themes). Idempotent.
 *
 * @returns The ECharts core namespace.
 */
export function loadEcharts(): typeof echarts {
  return echartsCore()
}
