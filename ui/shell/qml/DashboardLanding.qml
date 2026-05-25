import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root
    color: "#1A1A24"

    signal quickActionRequested(string actionId)

    ScrollView {
        anchors.fill: parent
        contentWidth: availableWidth

        ColumnLayout {
            width: root.width
            spacing: 24
            anchors.margins: 32

            Label {
                text: "Welcome back!"
                color: "#E8E8F0"
                font.pixelSize: 28
                font.bold: true
            }

            Label {
                text: "What would you like to work on today?"
                color: "#A0A0B8"
                font.pixelSize: 13
            }

            GridLayout {
                columns: 3
                columnSpacing: 12
                rowSpacing: 12

                Repeater {
                    model: [
                        { label: "Open Project", id: "open_project" },
                        { label: "New Project", id: "new_project" },
                        { label: "Quick OCR", id: "quick_image" },
                        { label: "Translate Folder", id: "batch_queue" },
                        { label: "Translation Assist", id: "assist_qa" },
                        { label: "Models / AI", id: "models_ai" }
                    ]
                    delegate: Rectangle {
                        Layout.preferredWidth: 160
                        Layout.preferredHeight: 90
                        radius: 12
                        color: mouseArea.containsMouse ? "#2A2A3D" : "#222233"
                        border.color: mouseArea.containsMouse ? "#7B5CFF" : "#33334A"
                        border.width: 1

                        Text {
                            anchors.fill: parent
                            text: modelData.label
                            color: "#E8E8F0"
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            font.pixelSize: 13
                            font.weight: Font.Medium
                        }

                        MouseArea {
                            id: mouseArea
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: root.quickActionRequested(modelData.id)
                        }
                    }
                }
            }
        }
    }
}
