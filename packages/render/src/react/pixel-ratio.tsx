/**
 * The ambient pixel ratio: `PixelRatioProvider` says by how much the figures below it are shown magnified (a slide
 * scaled to the window, a page-size figure in an enlarged stage), and `usePixelRatio` reads it. `Chart` takes it when
 * it has no `ratio` of its own, so a host sets it once instead of passing it to every chart.
 */

import { createContext, type ReactElement, type ReactNode, useContext } from 'react'

const PixelRatioContext = createContext<number | undefined>(undefined)

/**
 * Say by how much everything below is shown magnified, so its canvas charts are drawn at `devicePixelRatio x ratio` and
 * stay sharp at the size they are shown. Nestable: the innermost provider wins (it replaces, it does not multiply).
 * Renders no element. Nothing is clamped (a host that wants a cap, for memory, clamps the number it gives). SVG charts
 * (`renderer="svg"`) are vectors and ignore it. See {@link Chart} for how it combines with the `ratio` prop and an
 * expanded `FigureFrame`.
 *
 * @param props - `ratio`, the magnification (1 is the browser's own device pixel ratio), and the children.
 * @returns The provider.
 * @index function React component: set the pixel ratio magnification of the charts below: `<PixelRatioProvider ratio={1.6}>`
 * @example
 * ```tsx
 * <PixelRatioProvider ratio={windowWidth / 1600}>
 *   <Chart option={option} />
 * </PixelRatioProvider>
 * ```
 */
export function PixelRatioProvider(props: { readonly ratio: number; readonly children?: ReactNode }): ReactElement {
  return <PixelRatioContext.Provider value={props.ratio}>{props.children}</PixelRatioContext.Provider>
}

/**
 * The magnification the nearest {@link PixelRatioProvider} gives, or `undefined` with none.
 *
 * @returns The ratio, or `undefined`.
 * @index function React hook: the ambient pixel ratio magnification, if a host set one: `usePixelRatio()`
 */
export function usePixelRatio(): number | undefined {
  return useContext(PixelRatioContext)
}
