/**
 * The ECharts core the React bindings draw with: `loadEcharts` hands out the shared tree-shaken core from
 * `omai-render/charts`, with every chart type, both renderers and the omai themes.
 */

import type * as echarts from 'echarts/core'
import { echartsCore } from '../charts/echarts-core.ts'

/**
 * The ECharts core the React bindings draw with: `echartsCore` from `omai-render/charts` (every chart type and
 * component the builders use, both renderers, the omai themes). Idempotent.
 *
 * @returns The ECharts core namespace.
 */
export function loadEcharts(): typeof echarts {
  return echartsCore()
}
