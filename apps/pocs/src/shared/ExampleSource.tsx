import type { Config } from 'omai-engine/trace'
import { FigurePanel, type Panels } from 'omai-render/react'
import { type ReactNode, useEffect, useMemo, useState } from 'react'
import { CodeBlock } from '../../../site/src/shell/CodeBlock.tsx'
import { codeOf, type ExampleCode, sourcesOf } from './exampleCode.ts'

/** The panel name the example pages register for it. */
export const SOURCE_PANEL = 'source'

/**
 * The foot of an example page: the code that ran, a collapsible panel. The model's pieces for the current choice of the knobs
 * and the lines that choose, bind and run the inference method, all cut from the library's own source files by their `// region`
 * markers, so the code on screen is the code that runs. The sources load when the panel is open.
 *
 * @param props - The page's panels, its entry of {@link exampleCode}, its name, and the configuration.
 * @returns The panel.
 */
export function ExampleSource(props: {
  readonly panels: Panels
  readonly entry: ExampleCode
  readonly name: string
  readonly config: Config
}): ReactNode {
  return (
    <FigurePanel panels={props.panels} name={SOURCE_PANEL} title="The code that ran">
      <ExampleCodeBlock entry={props.entry} name={props.name} config={props.config} />
    </FigurePanel>
  )
}

/** The code block itself (for a page that has its own panels): loads the sources when it is mounted. */
export function ExampleCodeBlock(props: {
  readonly entry: ExampleCode
  readonly name: string
  readonly config: Config
}): ReactNode {
  const { entry, name, config } = props
  const [sources, setSources] = useState<Readonly<Record<string, string>> | undefined>()
  useEffect(() => {
    let live = true
    void sourcesOf(entry).then((files) => {
      if (live) setSources(files)
    })
    return () => {
      live = false
    }
  }, [entry])
  const shown = useMemo(
    () => (sources === undefined ? undefined : codeOf(entry, config, sources)),
    [entry, config, sources],
  )
  if (shown === undefined) return <p className="text-sm text-muted">Loading the source...</p>
  return (
    <CodeBlock
      code={shown.code}
      title={`${shown.file}${entry.inference === undefined ? ' and example-method.ts' : ''}: ${name}`}
      className="max-w-4xl"
    />
  )
}
