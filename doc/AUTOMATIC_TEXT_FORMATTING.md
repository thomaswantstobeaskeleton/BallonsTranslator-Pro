# Automatic text formatting and bubble fitting

BallonsTranslator-Pro now performs a more automatic lettering pass when Auto layout or Auto fit is used. The goal is to reduce manual work for font size, line breaks, text-box size, and bubble placement.

## What the automatic pass does

- **Shape-aware safe area:** round, diamond, narrow, pointed, and elongated bubbles use different inner insets so text avoids corners and curved edges. The bubble mask is also scanned for the largest high-coverage inner rectangle so curved/pointed balloons do not use unsafe transparent corners as text area.
- **Balanced line breaking:** the layout engine tries width candidates derived from bubble size, text length, line height, and bubble shape instead of only shrinking from the widest line. This helps avoid uneven lines and isolated short stubs.
- **Density-aware font scaling:** short phrases stay larger, while dense or multi-line translations shrink more aggressively only when needed.
- **Final rendered-fit polish:** after the text is broken into lines and the text box is placed, a final document-size check shrinks overflow, gently expands under-filled roomy boxes, and performs extra mask-corner shrinking for curved bubbles.
- **Less import-time fragility:** pure auto-layout heuristics live in `utils/auto_text_layout.py`, keeping tests and headless tooling away from image-runtime dependencies until rendering actually needs them.
- **Automatic binary-search fitting:** the default auto-layout profile now enables binary-search font fitting for non-CJK horizontal text, choosing the largest font size that fits the detected bubble.

## Settings

The most useful settings are in **Config Panel → Lettering**:

- **Auto lettering preset**: choose the overall automatic behavior without tuning many individual values. **Balanced** is recommended and adapts per bubble: dense/tiny/narrow bubbles become safer while short roomy bubbles get a readability boost. **Fit inside bubble** always uses stricter margins and smaller text; **Larger readable text** always allows wider lines and larger font when safe.
- **Scale font to fit bubble**: keeps laid-out text inside the detected bubble.
- **Binary search font size**: tries multiple font sizes and chooses the largest fitting size. It is more accurate and is enabled by default.
- **Final overflow safety pass**: performs a last rendered-size polish to prevent clipping and improve under-filled boxes. It is enabled by default.
- **Use mask-safe inner bubble area**: scans the detected bubble mask to avoid curved/pointed corners. It is enabled by default.
- **Balloon shape (Diamond-Text)**: leave this on **Auto** for automatic shape-specific margins and line scoring.

Existing projects remain compatible. Older projects that do not have the new final safety or mask-safe-area settings behave as if they are enabled.

## One-click bubble lettering polish

The **Format → Smart auto fit lettering** and **Format → Atomic bubble fit** actions now run a fuller safety pipeline instead of only changing the font size:

1. preserve spaces in Latin translations while balancing line lengths, so automatic rewrapping no longer glues words together;
2. choose script-aware line-breaking and writing mode before sizing;
3. apply the chosen font size/spacing to the selected text box;
4. grow the box modestly from renderer diagnostics if text still cannot fit at the allowed minimum size;
5. run a final rendered-size safety pass and, when a bubble mask is available, recenter the text box in its bubble.

This keeps everyday editing focused on one or two high-level actions instead of repeatedly tuning size, line spacing, padding, and position by hand. Existing manual controls remain available for special lettering or SFX cases.

## Simplified auto-layout profiles

The many legacy auto-layout knobs are now grouped behind the **Auto lettering preset** selector:

- **Balanced (recommended)**: model-free contour/aspect-ratio bubble detection, mask-safe inner areas, centered bubbles, optimal line breaks, binary font fitting, and final overflow safety. This is the default for most manga/manhua pages.
- **Fit inside bubble**: stricter height/short-line penalties, a lower minimum font size, compact widths, and strong one-word-line avoidance for crowded bubbles where staying inside the balloon matters most.
- **Larger readable text**: higher minimum/maximum font size, fewer-line optimization, and roomier widths while still keeping final overflow checks and bubble centering enabled.

Changing the preset applies recommended values to the advanced controls below it. The **Reset advanced values to selected preset** button can also normalize older projects that already saved confusing manual values. Optional model fields are blank by default; geometry and mask checks are used unless you explicitly enter a model ID.

## Reading the advanced numbers

Advanced numeric controls now show a live **What the advanced values mean** summary in Settings. Instead of requiring you to memorize scales such as `80`, `360`, or `2000`, the panel translates them into plain-language behavior:

- short-line and one-word penalties are labeled loose/balanced/strict/maximum;
- height overflow is labeled roomy, balanced, strict fit, or very strict fit;
- line-width controls are labeled compact, balanced, roomy, or wide;
- font-size min/max is summarized as a readable point-size range;
- model fields say whether layout is using geometry only, built-in CLIP, a custom model, or model-assisted shape detection.

If the settings feel confusing, use **Restore balanced automatic defaults**. It resets only the advanced auto-layout values and leaves unrelated text style, font, color, and rendering defaults alone.
