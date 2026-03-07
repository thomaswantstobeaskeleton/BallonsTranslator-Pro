# Photoshop scripts for BallonsTranslator

Scripts in this folder are run from **Adobe Photoshop** (ExtendScript). Drag the `.jsx` file into a Photoshop window, or use **File → Scripts → Browse...** and select the script.

## Load_png_into_PSD.jsx

Batch merge BallonsTranslator PNG masks with PSD files for AI manga whitening:

1. Choose the folder containing your **PSD** files (one per page).
2. Choose the folder containing **PNG** files (e.g. BallonsTranslator mask or result exports) with the **same base name** as each PSD (e.g. `001.psd` and `001.png`).
3. Click OK. For each PSD, the script opens it, places the matching PNG as a new layer, sets the layer blend mode to **Screen**, and saves the PSD.

Use case: combine translated/inpainted PNGs with layered PSDs so you can further edit or feed them to a model for whitening.
