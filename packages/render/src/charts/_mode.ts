// Internal: layers built without a colour mode, redrawn in the mode the chart is composed in.

import type { ColourMode } from '../theme/tokens.ts'
import type { Layer } from './plot.ts'

// How to build a layer again in another mode, for the layers that were built without one. Kept beside the layer rather
// than on it, so a layer that is copied or edited (`{ ...l, secondary: true }`) is plain data again and never carries
// a recipe that would redraw it as it was before the edit.
const recipes = new WeakMap<object, (mode: ColourMode) => Layer>()

/**
 * Whether options leave the colour mode open: no `mode`, and no `figure` (whose mode would stand for it).
 *
 * @param options - A builder's options.
 * @returns True when the builder is free to take the mode of the chart it is composed in.
 */
export function isModeless(options: { readonly mode?: ColourMode | undefined; readonly figure?: unknown }): boolean {
  return options.mode === undefined && options.figure === undefined
}

/**
 * Build a layer in the light mode and remember how to build it in the other. The layer is what the builder returns
 * today; composing it in another mode (`atMode`) calls the builder again with that mode, so the result is exactly the
 * layer built with the mode given, whatever the builder derives its colours from.
 *
 * @param build - The builder, called with the mode to build in.
 * @returns The layer built in the light mode.
 */
export function deferMode<L extends Layer>(build: (mode: ColourMode) => L): L {
  const cache = new Map<ColourMode, L>()
  const at = (mode: ColourMode): L => {
    let built = cache.get(mode)
    if (built === undefined) {
      built = build(mode)
      cache.set(mode, built)
    }
    return built
  }
  const light = at('light')
  recipes.set(light, at)
  return light
}

/**
 * {@link deferMode} for a builder of several layers: each layer remembers its own place in the list the builder
 * returns for a mode (the builder runs once per mode).
 *
 * @param build - The builder, called with the mode to build in.
 * @returns The layers built in the light mode.
 */
export function deferModes(build: (mode: ColourMode) => Layer[]): Layer[] {
  const cache = new Map<ColourMode, Layer[]>()
  const at = (mode: ColourMode): Layer[] => {
    let built = cache.get(mode)
    if (built === undefined) {
      built = build(mode)
      cache.set(mode, built)
    }
    return built
  }
  const light = at('light')
  light.forEach((l, i) => {
    recipes.set(l, (mode) => at(mode)[i] as Layer)
  })
  return light
}

/**
 * A layer as it is in a colour mode: one built without a mode is built again in `mode` (exactly what the builder gives
 * when asked for that mode); any other layer (built with a `mode`, a `figure`, or edited since) is returned as it is.
 *
 * @param layer - A layer.
 * @param mode - The mode of the chart that draws it.
 * @returns The layer for the mode.
 */
export function atMode<L extends Layer>(layer: L, mode: ColourMode): L {
  if (mode === 'light') return layer
  const recipe = recipes.get(layer)
  return recipe === undefined ? layer : (recipe(mode) as L)
}
