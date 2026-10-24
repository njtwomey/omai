import { type ReactNode, useMemo, useState } from 'react'
import { iris } from 'whitebox-engine/datasets'
import { toArray } from 'whitebox-engine/tensor'
import { type ScatterMatrixCell, scatterMatrix, scatterMatrixCells, type Table } from 'whitebox-render/charts'
import { brushSelection, type Figure, figure } from 'whitebox-render/figure'
import {
  Brush,
  Chart,
  Choice,
  ControlPanel,
  type DataRect,
  FigureFrame,
  InteractionOverlay,
  linearProjection,
  useColourMode,
} from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Scatter matrix',
  question: 'How do I plot every pair of columns, and brush in one panel to select everywhere?',
  explain:
    '`scatterMatrix` draws every pair of columns, with each column’s histogram and name on the diagonal. `scatterMatrixCells` says where each panel sits and what it spans, so a `Brush` over any panel (through a `linearProjection` of its ranges) selects rows, and `selection` lights those rows up in every panel.',
}

const data = iris()
const table: Table = {
  columns: data.featureNames.map((f) => f.replace(/ \(cm\)$/, '')),
  rows: toArray(data.x) as number[][],
  groups: Array.from(data.y.data),
  groupNames: Object.fromEntries(data.classes.map((c, i) => [i, c])),
}
const PADDING = { top: 28, right: 16 }
const FIG = figure({ width: 560, aspect: 1, padding: PADDING })
const panelsOf = (fig: Figure): ScatterMatrixCell[] =>
  scatterMatrixCells(table, { figure: fig }).filter((c) => !c.diagonal)

export default function ScatterMatrixRecipe(): ReactNode {
  const mode = useColourMode()
  const [diagonal, setDiagonal] = useState<'histogram' | 'label'>('histogram')
  const [brush, setBrush] = useState<{ readonly cell: string; readonly rect: DataRect } | null>({
    cell: 'r3c2',
    rect: { x: [4.5, 7], y: [1.3, 2.6] },
  })
  const selection = useMemo(() => {
    if (brush === null) return undefined
    // Panel r{row}c{col} plots column col across and column row up.
    const [, row = 0, col = 0] = /r(\d+)c(\d+)/.exec(brush.cell)?.map(Number) ?? []
    const [i, j] = [col, row]
    return brushSelection(
      table.rows.map((r) => [r[i] ?? 0, r[j] ?? 0] as const),
      brush.rect,
    )
  }, [brush])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={1}>
        <Choice label="diagonal" value={diagonal} options={['histogram', 'label'] as const} onChange={setDiagonal} />
      </ControlPanel>
      <FigureFrame
        figure={FIG}
        filename="scatter-matrix"
        caption={`${selection?.length ?? table.rows.length} of ${table.rows.length} flowers selected. Drag in any panel to brush; click to clear.`}
      >
        {(size) => {
          // The same figure at the frame's size (larger when expanded), and its panels for the brushes.
          const fig = figure({ width: size.width, height: size.height, padding: PADDING })
          const option = scatterMatrix(table, {
            mode,
            figure: fig,
            diagonal,
            ...(selection === undefined ? {} : { selection }),
          })
          return (
            <div style={{ position: 'relative', width: fig.width, height: fig.height }}>
              <Chart option={option} figure={fig} />
              {panelsOf(fig).map((c) => (
                <div
                  key={c.id}
                  style={{
                    position: 'absolute',
                    left: c.rect.x,
                    top: c.rect.y,
                    width: c.rect.width,
                    height: c.rect.height,
                  }}
                >
                  <InteractionOverlay
                    projection={linearProjection({
                      x: c.xRange,
                      y: c.yRange,
                      width: c.rect.width,
                      height: c.rect.height,
                    })}
                    width={c.rect.width}
                    height={c.rect.height}
                  >
                    <Brush
                      value={brush?.cell === c.id ? brush.rect : null}
                      onChange={(rect) => setBrush(rect === null ? null : { cell: c.id, rect })}
                    />
                  </InteractionOverlay>
                </div>
              ))}
            </div>
          )
        }}
      </FigureFrame>
    </div>
  )
}
