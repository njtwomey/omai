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
