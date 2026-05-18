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
