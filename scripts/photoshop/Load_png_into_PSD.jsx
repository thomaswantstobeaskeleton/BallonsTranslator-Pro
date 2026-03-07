/*
 * Load PNG into PSD.js
 * From https://github.com/jqk4388/PS-Script-BallonsTranslator
 *
 * Stacks PSD and PNG files with the same base name: opens each PSD,
 * places the matching PNG as a layer, sets blend mode to "Screen",
 * saves the PSD. Use with BallonsTranslator mask PNGs and PSDs for
 * AI manga whitening workflows. Drag this script into a Photoshop
 * window or run via File → Scripts → Browse....
 */

#target photoshop

var psdFolder, pngFolder;

var dialog = new Window("dialog", "Batch Stack PSD and PNG");

var psdGroup = dialog.add("group");
psdGroup.add("statictext", undefined, "PSD Folder:");
var psdInput = psdGroup.add("edittext", undefined, "");
psdInput.size = [300, 25];
var psdBrowse = psdGroup.add("button", undefined, "Browse");

psdBrowse.onClick = function () {
    psdFolder = Folder.selectDialog("Select PSD folder");
    if (psdFolder) psdInput.text = psdFolder.fsName;
};

var pngGroup = dialog.add("group");
pngGroup.add("statictext", undefined, "PNG Folder:");
var pngInput = pngGroup.add("edittext", undefined, "");
pngInput.size = [300, 25];
var pngBrowse = pngGroup.add("button", undefined, "Browse");

pngBrowse.onClick = function () {
    pngFolder = Folder.selectDialog("Select PNG folder");
    if (pngFolder) pngInput.text = pngFolder.fsName;
};

var buttonGroup = dialog.add("group");
buttonGroup.alignment = "center";
buttonGroup.add("button", undefined, "OK", { name: "ok" });
buttonGroup.add("button", undefined, "Cancel", { name: "cancel" });

if (dialog.show() != 1) exit();

if (!psdFolder || !pngFolder) {
    alert("Both PSD and PNG folders must be selected.");
    exit();
}

var psdFiles = psdFolder.getFiles("*.psd");
var pngFiles = pngFolder.getFiles("*.png");

if (psdFiles.length === 0 || pngFiles.length === 0) {
    alert("No PSD or PNG files found in the selected folders.");
    exit();
}

for (var i = 0; i < psdFiles.length; i++) {
    var psdFile = psdFiles[i];
    var baseName = psdFile.name.replace(/\.psd$/i, "");
    var matchingPng;
    for (var j = 0; j < pngFiles.length; j++) {
        if (pngFiles[j].name.replace(/\.png$/i, "") === baseName) {
            matchingPng = pngFiles[j];
            break;
        }
    }

    if (matchingPng) {
        try {
            var doc = app.open(psdFile);
            var pngDoc = app.open(matchingPng);
            pngDoc.activeLayer.duplicate(doc);
            pngDoc.close(SaveOptions.DONOTSAVECHANGES);
            doc.activeLayer.blendMode = BlendMode.SCREEN;
            var saveOptions = new PhotoshopSaveOptions();
            doc.saveAs(psdFile, saveOptions, true, Extension.LOWERCASE);
            doc.close(SaveOptions.DONOTSAVECHANGES);
        } catch (e) {
            alert("Error processing files: " + psdFile.name + " and " + matchingPng.name + "\n" + e.message);
        }
    }
}
