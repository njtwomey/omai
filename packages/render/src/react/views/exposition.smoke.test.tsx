import { type Exposition, type ExpositionKind, expositionSamples } from 'omai-engine/exposition'
import type { ReactElement } from 'react'
import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { viewFor } from '../../views/registry.ts'
import { mountElement, resetGlobals } from '../_harness.tsx'
import { ExpositionView } from './ExpositionView.tsx'
import * as views from './index.ts'
import { RecordBoundary } from './RecordBoundary.tsx'

const kinds = Object.keys(expositionSamples) as ExpositionKind[]

// A pipeline whose stage shows a table has no frame to title, and a threshold sweep is several figures: both leave the
// title to the page. Every other view puts the title it is given above its figure.
const UNTITLED: readonly ExpositionKind[] = ['pipelineStages', 'thresholdSweep']

function Bad(): ReactElement {
  throw new Error('this record cannot be drawn')
}

describe('ExpositionView over a small valid record of every kind', () => {
  it('draws every kind on the server, through a component the registry names and the package exports', () => {
    for (const kind of kinds) {
      const record = expositionSamples[kind]() as Exposition
      const html = renderToString(<ExpositionView record={record} />)
      expect(html.length, kind).toBeGreaterThan(0)
      expect(
        renderToString(<ExpositionView record={record} title="A TITLE" />).includes('A TITLE'),
        `${kind}: title`,
      ).toBe(!UNTITLED.includes(kind))
      expect(Object.keys(views), kind).toContain(viewFor(kind).component)
    }
  })

  it('says so for a kind it does not know, and one bad record does not take the page with it', async () => {
    const html = renderToString(<ExpositionView record={{ kind: 'nope' } as never} />)
    expect(html).toContain('no standard view for record kind')
    expect(html).toContain('nope')
    const record = expositionSamples.curves()
    await expect(mountElement(<Bad />)).rejects.toThrow('this record cannot be drawn')
    const kept = await mountElement(
      <RecordBoundary record={record}>
        <Bad />
      </RecordBoundary>,
    )
    await kept.unmount()
    resetGlobals()
  })
})
