import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root
    width: 200
    color: "#121218"

    property string currentSection: "home"
    signal navigateRequested(string sectionId)

    readonly property var sections: [
        { id: "home", label: "Home" },
        { id: "editor", label: "Editor" },
        { id: "live_translate", label: "Live Translate" },
        { id: "quick_image", label: "Quick Image" },
        { id: "downloader", label: "Downloader" },
        { id: "batch_queue", label: "Batch Queue" },
        { id: "assist_qa", label: "Assist / QA" },
        { id: "models_ai", label: "Models / AI" },
        { id: "settings", label: "Settings" },
        { id: "diagnostics", label: "Diagnostics" }
    ]

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 8
        spacing: 4

        Label {
            text: "B"
            color: "#7B5CFF"
            font.pixelSize: 18
            font.bold: true
            Layout.fillWidth: true
            Layout.preferredHeight: 48
            verticalAlignment: Text.AlignVCenter
            leftPadding: 12
        }

        Repeater {
            model: root.sections
            delegate: Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 40
                radius: 8
                color: modelData.id === root.currentSection ? "rgba(123, 92, 255, 0.18)" : (mouseArea.containsMouse ? "#222233" : "transparent")

                Text {
                    anchors.fill: parent
                    anchors.leftMargin: 12
                    text: modelData.label
                    color: modelData.id === root.currentSection ? "#9B80FF" : "#A0A0B8"
                    font.pixelSize: 13
                    font.weight: modelData.id === root.currentSection ? Font.DemiBold : Font.Medium
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignLeft
                }

                MouseArea {
                    id: mouseArea
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: root.navigateRequested(modelData.id)
                }
            }
        }
        Item { Layout.fillHeight: true }
    }
}
