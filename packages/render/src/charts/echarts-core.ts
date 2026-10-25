/**
 * The tree-shaken ECharts core: every chart type and component the `whitebox-render/charts` builders use, the SVG and
 * canvas renderers, and the whitebox light and dark themes registered, shared by the React bindings and server-side SVG
 * export.
 */

import {
  BarChart,
  BoxplotChart,
  CustomChart,
  HeatmapChart,
  LineChart,
  LinesChart,
  ParallelChart,
  RadarChart,
  ScatterChart,
} from 'echarts/charts'
import {
  AxisPointerComponent,
  DataZoomInsideComponent,
  DataZoomSliderComponent,
  GraphicComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  MarkPointComponent,
  ParallelComponent,
  PolarComponent,
  RadarComponent,
  TitleComponent,
  TooltipComponent,
  VisualMapContinuousComponent,
  VisualMapPiecewiseComponent,
} from 'echarts/components'
import * as echarts from 'echarts/core'
import { CanvasRenderer, SVGRenderer } from 'echarts/renderers'
import { echartsTheme, themeName } from '../theme/echarts.ts'

/**
 * The tree-shaken ECharts core with every chart type and component the `whitebox-render/charts` builders use, the SVG
 * and canvas renderers, and the whitebox light and dark themes registered. Idempotent; works in Node (server-side
 * rendering for export) and the browser.
 *
 * @returns The ECharts core namespace.
 * @example
 * ```ts
 * import { echartsCore } from 'whitebox-render/charts'
 * typeof echartsCore().init // => 'function'
 * ```
 */
export function echartsCore(): typeof echarts {
  echarts.use([
    BarChart,
    BoxplotChart,
    CustomChart,
    HeatmapChart,
    LineChart,
    LinesChart,
    ParallelChart,
    RadarChart,
    ScatterChart,
    AxisPointerComponent,
    DataZoomInsideComponent,
    DataZoomSliderComponent,
    GraphicComponent,
    GridComponent,
    LegendComponent,
    MarkAreaComponent,
    MarkLineComponent,
    MarkPointComponent,
    ParallelComponent,
    PolarComponent,
    RadarComponent,
    TitleComponent,
    TooltipComponent,
    VisualMapContinuousComponent,
    VisualMapPiecewiseComponent,
    CanvasRenderer,
    SVGRenderer,
  ])
  for (const mode of ['light', 'dark'] as const) echarts.registerTheme(themeName(mode), echartsTheme(mode))
  return echarts
}
