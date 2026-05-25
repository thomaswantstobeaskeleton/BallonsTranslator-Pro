import QtQuick
import QtQuick.Controls

Label {
    id: root
    property color pillColor: "#4ADE80"
    color: "#121218"
    font.pixelSize: 10
    font.bold: true
    padding: 6
    leftPadding: 10
    rightPadding: 10
    background: Rectangle {
        radius: 999
        color: root.pillColor
    }
}
