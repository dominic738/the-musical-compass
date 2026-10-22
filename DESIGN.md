---
name: The Musical Compass
description: A political compass for songs, placed by the meaning of their lyrics.
colors:
  quadrant-red: "#FF7575"
  quadrant-blue: "#42AAFF"
  quadrant-green: "#9AED97"
  quadrant-violet: "#C09AEC"
  ink: "#000000"
  ink-soft: "#4D4D4D"
  muted: "#6B6B6B"
  paper: "#FFFFFF"
  sidebar-grey: "#F2F2F2"
  sidebar-rule: "#CFCFCF"
  field-rule: "#9A9A9A"
  grid-line: "rgba(0, 0, 0, 0.16)"
  danger: "#C40000"
  dot-red: "#E10600"
  dot-black: "#111111"
  dot-white: "#FFFFFF"
  dot-yellow: "#FFD400"
  dot-orange: "#FF7A00"
  dot-navy: "#1638A8"
typography:
  display:
    fontFamily: "Arimo, Arial, Helvetica, sans-serif"
    fontSize: "30px"
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.02em"
    fontFeature: "\"tnum\""
  headline:
    fontFamily: "Arimo, Arial, Helvetica, sans-serif"
    fontSize: "22px"
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.02em"
  pole:
    fontFamily: "Arimo, Arial, Helvetica, sans-serif"
    fontSize: "20px"
    fontWeight: 700
  title:
    fontFamily: "Arimo, Arial, Helvetica, sans-serif"
    fontSize: "18px"
    fontWeight: 700
  section:
    fontFamily: "Arimo, Arial, Helvetica, sans-serif"
    fontSize: "15px"
    fontWeight: 700
  body:
    fontFamily: "Arimo, Arial, Helvetica, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.45
    fontFeature: "\"tnum\""
  body-small:
    fontFamily: "Arimo, Arial, Helvetica, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.45
  label:
    fontFamily: "Arimo, Arial, Helvetica, sans-serif"
    fontSize: "12.5px"
    fontWeight: 700
rounded:
  none: "0px"
  round: "50%"
spacing:
  hair: "4px"
  xs: "6px"
  sm: "8px"
  md: "10px"
  lg: "16px"
  side-y: "22px"
  side-x: "26px"
  stage-x: "40px"
  sidebar-width: "360px"
components:
  button-primary:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.section}"
    rounded: "{rounded.none}"
    padding: "8px 14px"
    width: "100%"
  button-primary-hover:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
  button-primary-disabled:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.muted}"
  button-secondary:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.body-small}"
    rounded: "{rounded.none}"
    padding: "7px 12px"
  button-secondary-hover:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
  button-quiet:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "8px 14px"
  link-button:
    textColor: "{colors.ink}"
    typography: "{typography.body-small}"
    padding: "0"
  input-text:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.none}"
    padding: "7px 9px"
    width: "100%"
  sidebar-panel:
    backgroundColor: "{colors.sidebar-grey}"
    textColor: "{colors.ink}"
    padding: "22px 26px"
  song-row:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    typography: "{typography.body-small}"
    padding: "6px 4px 6px 8px"
  song-row-selected:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
  flair-chip:
    rounded: "{rounded.none}"
    size: "12px"
  flair-chip-result:
    rounded: "{rounded.none}"
    size: "16px"
  dot-swatch:
    rounded: "{rounded.round}"
    size: "24px"
  dialog:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "26px 28px"
    width: "500px"
  build-card:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "20px 22px"
    width: "420px"
  chart-notice:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.section}"
    padding: "10px 14px"
  toast:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    typography: "{typography.body-small}"
    rounded: "{rounded.none}"
    padding: "9px 14px"
  toast-error:
    backgroundColor: "{colors.danger}"
    textColor: "{colors.paper}"
---

# Design System: The Musical Compass

## Overview

**Creative North Star: "The Quadrant Meme, Played Straight"**

The whole system is the political compass chart, rebuilt with care rather than parodied. A white page, a square chart with four flat pastel quadrants, a faint unit grid, a hard black frame and black axes, bold Arial-lineage lettering outside each edge. Every other surface is a plain utility page around it: a grey sidebar of forms on the left, black-bordered buttons, square inputs. Nothing is rounded except the things that are dots.

Density is that of a worksheet: compact 15px text, tight 10px field gaps, the chart as large as the viewport allows. The interface refuses the card-dashboard idiom (rounded panels, soft neutral greys, one brand accent, drop shadows). Colour comes only from the four quadrants and the user's dot colours; the chrome is black, white and grey.

The signature is the result readout under the chart: each song is reported like a political compass test result, two signed scores on the −10 to +10 scale in large tabular figures and a meme-style quadrant flair ("AngrySad") with a square chip in its quadrant colour.

**Key Characteristics:**
- Four quadrant hues in their political-compass positions, used flat and at full strength.
- Black 2px frame, black 2px axes, 1px grid at 16% black, one line per scale unit (20×20).
- One typeface (Arimo, metric-compatible with Arial), bold for every label, tabular figures throughout.
- Square corners on all chrome; circles only for data dots, colour swatches and spinners.
- Flat: no elevation system. Depth is drawn with black borders.

## Colors

A chart palette, not a brand palette: four pastel quadrant fills carry all the colour, set against black ink, white paper and utility greys.

### Primary
- **Quadrant Red** (quadrant-red): top-left quadrant fill; also the error state of progress rows and the custom-swatch wheel.
- **Quadrant Blue** (quadrant-blue): top-right quadrant fill; the in-progress fill of build progress bars and the text-selection highlight.
- **Quadrant Green** (quadrant-green): bottom-left quadrant fill; the completed state of build progress rows.
- **Quadrant Violet** (quadrant-violet): bottom-right quadrant fill.

The same four hues fill the flair chips in the song list, the result readout, the exported legend and the brand mark (a 2×2 of the four squares with black cross and frame).

### Secondary
- **Dot Palette** (dot-red, dot-black, dot-white, dot-yellow, dot-orange, dot-navy): the six preset fills for song dots, plus a free custom colour. Saturated and dark so a dot reads against any pastel quadrant; every dot carries a black stroke. Red is the default.

### Neutral
- **Ink** (ink): all text, the chart frame and axes, every button border, focus rings, selected-row border, toast background.
- **Soft Ink** (ink-soft): secondary text: artist names, score captions, dialog body copy, lede.
- **Muted Grey** (muted): placeholders, hints, footer, disabled text, pending pole words.
- **Paper** (paper): page ground, chart surround, inputs, dialogs, selected song row.
- **Sidebar Grey** (sidebar-grey): the utility sidebar ground.
- **Sidebar Rule** (sidebar-rule): 1px dividers between sidebar sections and in the build card.
- **Field Rule** (field-rule): 1px input borders, disabled button borders, idle progress squares.
- **Grid Line** (grid-line): the unit grid inside the chart.
- **Danger** (danger): error toast ground only.

### Named Rules
**The Four Corners Rule.** Red top-left, blue top-right, green bottom-left, violet bottom-right, at exactly these values. The quadrant hues are the reference; they are never tinted, swapped or joined by a fifth accent.

**The Colour Plus Word Rule.** A quadrant is never identified by colour alone: every chip sits beside its flair name, and every quadrant prints its flair in its corner.

## Typography

**Display Font:** Arimo (with Arial, Helvetica, sans-serif)
**Body Font:** Arimo (same stack)

**Character:** Chart lettering. One Arial-lineage sans, self-hosted in four subsets (latin, latin-ext, cyrillic, greek) so non-Latin titles render in the same face. Weight does all the hierarchy work: bold for labels and titles, regular for running text.

### Hierarchy
- **Display** (700, 30px, 1.1, −0.02em): the two signed scores in the result readout. Signs use a true minus (−) and an explicit plus.
- **Headline** (700, 22px, 1.1, −0.02em): the wordmark and dialog titles (wordmark drops to 20px under 420px).
- **Pole** (700, 20px at screen size): the four pole words outside the chart edges; side poles rotate ±90°. Pending (unbuilt) words drop to muted grey.
- **Title** (700, 18px): the song name in the readout (artist follows in regular soft ink), the build card title.
- **Section** (700, 15px): sidebar step headings ("1. Axes"), buttons, the chart notice, the readout flair.
- **Body** (400, 15px, 1.45): inputs and default text. Dialog copy caps at 65ch, the lede at 34ch.
- **Body Small** (400, 14px): song rows, hints, secondary buttons, toasts.
- **Label** (700, 12.5px): field labels and the quadrant flair names in chart corners (at 82% black). Song scores and footer use the same size at regular weight.

### Named Rules
**The One Face Rule.** Arimo everywhere, including the chart's SVG text and the exported PNG (the font is embedded into the export). No second family, no display face.

**The Tabular Figures Rule.** `tabular-nums` is set on the body; every score lines up digit for digit.

**The Screen-Size Chart Type Rule.** Chart text, dots and strokes are multiplied by a scale variable so they keep their screen pixel size however large the chart is drawn.

## Layout

A two-column app shell on a CSS grid: a fixed 360px sidebar on the left holding four stacked panels (masthead, 1. Axes, 2. Add a song, Songs) and a footer; the stage on the right holds the chart and the result readout, centred and sticky to the top of the viewport.

- **Chart size:** the square chart and readout share one width, `max(320px, min(800px, 100vh − 175px))`, so the whole chart plus readout fits in one desktop viewport. The SVG has an 804×796 viewBox around a 700-unit plot square; overlays (empty notice, build progress) are inset to the plot square by percentage (6.47% / 6.03%).
- **Grid:** 20×20 units, one line per point on the −10 to +10 scale; the centre lines are the axes.
- **Sidebar rhythm:** panels pad 22px by 26px, separated by 1px rules; fields stack with 10px gaps; axis inputs sit in a two-column grid (Up/Down, Left/Right) with 10px gutter.
- **Readout:** padded in by the same 6.47% so it aligns with the plot edge; scores sit side by side with a 36px gap; Save image sits to the right.
- **Under 900px:** one column in the order masthead, Axes, chart, Add a song, Songs, footer. The chart sits on white between grey panels, caps at 640px, loses stickiness; sidebar panels pad 18px by 16px; the song list grows without its own scroll. Adding a song scrolls the chart into view.
- **Under 420px:** Save image goes full width.
- The song list scrolls internally on desktop (max 340px).

## Elevation & Depth

Flat. Surfaces sit at one level and are separated by colour (grey sidebar against white stage) and by black borders: 2px around the chart, dialogs, the build card and the chart notice; 1px around inputs and the selected row. Dialogs rise over a 40% black backdrop, not on a shadow.

### Shadow Vocabulary
- **Toast** (`box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2)`): the only shadow in the build, on transient toasts. It is a lone exception, not a pattern to extend.

### Named Rules
**The Drawn Border Rule.** Anything that floats over the chart (notice, build card, dialog) is a white box with a 2px black border. Depth is drawn, never blurred.

## Shapes

Square. Every piece of chrome has a 0 radius: buttons, inputs, textareas, dialogs, the build card, song rows, flair chips, progress tracks. Circles are reserved for things that are points or paint: song dots, colour swatches (with a 2px ring 5px out when selected) and loading spinners. Line icons are 16px inline SVG with a 1.6 stroke and square caps.

## Components

### Buttons
Plain chart-page buttons, black on white.
- **Shape:** square (0 radius), 2px black border.
- **Primary:** white ground, black bold 15px text, 8px by 14px padding, full width in the sidebar (Generate compass, Add song, Place song).
- **Hover / Active:** inverts to black ground and white text over 0.12s; presses down 1px on active. Focus is a 2px black outline offset 2px.
- **Disabled:** grey border and muted text, not-allowed cursor.
- **Loading:** stays black-on-white with a 12px spinning ring before the label; progress cursor.
- **Secondary:** 1px border, regular weight 14px, 7px by 12px, with a leading 16px icon (Save image).
- **Quiet:** transparent border, 7% black ground on hover (dialog Cancel).
- **Link button:** underlined 13.5px text, offset 3px (How it works, Clear all).

### Inputs / Fields
- **Style:** white, 1px field-rule border, square, 7px by 9px, 15px text; bold 12.5px label above.
- **Hover:** border darkens to soft ink.
- **Focus:** border goes black and gains an inset 1px black ring (a 2px black edge), no outline glow.
- **Textarea:** same treatment, 200px minimum, vertical resize, 14px text.

### Dot Swatches
24px circles with a 1.5px black stroke; the selected one gets a 2px black ring 5px out. The last swatch is a conic wheel of the four quadrant hues that opens a colour picker and takes the chosen colour once set.

### Song List
Rows on the grey sidebar: a 12px flair chip, bold title then soft-ink artist (ellipsised), compact signed scores, a 26px remove button. Hover adds 4% black; the selected row turns white with a 1px black border. Hovering a row previews its result in the readout without changing the selection.

### Flair Chip
A small square in the quadrant colour with a 1.5px black border (12px in lists, 16px with a 2px border in the readout, 22px in the export legend), always beside the flair name.

### Dialogs
White, 2px black border, square, max 500px, 26px by 28px padding, over a 40% black backdrop. Headline-size title, soft-ink body, a 32px close icon top right, actions right-aligned (quiet Cancel, primary submit). Enter in with a 0.22s fade-and-rise of 8px; leave with a 0.15s fade.

### Toasts
Black ground, white 14px text, 9px by 14px, bottom centre; errors switch to danger red. They rise in like dialogs and fade out over 0.2s.

### The Compass (signature)
The SVG chart: four quadrant rects, 19 grid lines each way minus the centre, black axes and frame at a constant 2px screen stroke, bold pole words outside each edge, flair names in each corner. Song dots are 6.5px circles in the chosen colour with a 1.5px black stroke (8.5px and 2.5px when selected); labels are 12.5px with a 3.5px white halo, flip to the other side near the right edge, and nudge vertically to avoid collisions, with a hairline leader back to the dot when moved. A new dot drops from the centre to its place over 700ms on an exponential ease-out.

### The Result Readout (signature)
Under the chart: song title (bold) and artist, then two score blocks, each a bold soft-ink caption ("Sad ↔ Happy") over a 30px signed score, then the flair chip and name. Empty, it shows a muted hint.

### Build Progress
Drawn on the chart while a compass generates: a 2px-bordered white card listing the four words and two axes. Each row has a 14px square status mark (idle grey outline, spinning ring while loading, green fill when complete, red on error) and a 6px black-outlined track whose bar creeps to 85% in quadrant blue over 8s, then snaps to full green.

### Exported Image
A 1600px-wide PNG on white: the chart with embedded Arimo, a legend table under a 2px black rule (chip, title, artist, both scores, flair, selected row in bold), and a footer with the wordmark and site address.

## Do's and Don'ts

### Do:
- **Do** keep the four quadrant hues exact and in their political-compass corners.
- **Do** pair every quadrant colour with its flair name, in the chart and in every list.
- **Do** draw edges in black: 2px for the chart, buttons and floating boxes, 1px for inputs and selected rows.
- **Do** set every label in bold Arimo and every number in tabular figures with a true minus sign.
- **Do** size chart text and marks in screen pixels via the scale variable, so the chart reads the same at any size.
- **Do** honour reduced motion: animations and transitions collapse to near zero and the dot drop becomes instant.

### Don't:
- **Don't** round chrome; circles are only for dots, swatches and spinners.
- **Don't** add shadows to panels, cards or dialogs; the toast shadow is the single exception.
- **Don't** introduce a brand accent or a second typeface; colour belongs to the quadrants and the user's dots.
- **Don't** tint, gradient or soften the quadrant fills (the custom-swatch wheel is the only gradient).
