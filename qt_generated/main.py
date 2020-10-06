# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(761, 543)
        MainWindow.setMinimumSize(QtCore.QSize(500, 425))
        MainWindow.setMaximumSize(QtCore.QSize(1000, 756))
        MainWindow.setFocusPolicy(QtCore.Qt.ClickFocus)
        MainWindow.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.img_lbl = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.img_lbl.sizePolicy().hasHeightForWidth())
        self.img_lbl.setSizePolicy(sizePolicy)
        self.img_lbl.setAutoFillBackground(False)
        self.img_lbl.setText(_fromUtf8(""))
        self.img_lbl.setPixmap(QtGui.QPixmap(_fromUtf8(":/images/images/init.png")))
        self.img_lbl.setScaledContents(True)
        self.img_lbl.setObjectName(_fromUtf8("img_lbl"))
        self.verticalLayout_2.addWidget(self.img_lbl)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.prob_lineEdit = QtGui.QLineEdit(self.centralwidget)
        self.prob_lineEdit.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.prob_lineEdit.setObjectName(_fromUtf8("prob_lineEdit"))
        self.horizontalLayout.addWidget(self.prob_lineEdit)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout.addWidget(self.label_2)
        self.cmComboBox = QtGui.QComboBox(self.centralwidget)
        self.cmComboBox.setObjectName(_fromUtf8("cmComboBox"))
        self.horizontalLayout.addWidget(self.cmComboBox)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 761, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        self.menuEdit = QtGui.QMenu(self.menubar)
        self.menuEdit.setObjectName(_fromUtf8("menuEdit"))
        MainWindow.setMenuBar(self.menubar)
        self.toolBar = QtGui.QToolBar(MainWindow)
        self.toolBar.setMovable(True)
        self.toolBar.setAllowedAreas(QtCore.Qt.AllToolBarAreas)
        self.toolBar.setObjectName(_fromUtf8("toolBar"))
        MainWindow.addToolBar(QtCore.Qt.LeftToolBarArea, self.toolBar)
        self.actionLoad = QtGui.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/images/load.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionLoad.setIcon(icon)
        self.actionLoad.setObjectName(_fromUtf8("actionLoad"))
        self.actionEraser = QtGui.QAction(MainWindow)
        self.actionEraser.setEnabled(False)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/images/eraser.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionEraser.setIcon(icon1)
        self.actionEraser.setObjectName(_fromUtf8("actionEraser"))
        self.actionNext_Image = QtGui.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/images/next.jpg")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionNext_Image.setIcon(icon2)
        self.actionNext_Image.setObjectName(_fromUtf8("actionNext_Image"))
        self.actionPrevious_Image = QtGui.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/images/previous.jpg")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPrevious_Image.setIcon(icon3)
        self.actionPrevious_Image.setObjectName(_fromUtf8("actionPrevious_Image"))
        self.actionUndo = QtGui.QAction(MainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/images/undo.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionUndo.setIcon(icon4)
        self.actionUndo.setObjectName(_fromUtf8("actionUndo"))
        self.menuFile.addAction(self.actionLoad)
        self.menuEdit.addAction(self.actionNext_Image)
        self.menuEdit.addAction(self.actionPrevious_Image)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.toolBar.addAction(self.actionLoad)
        self.toolBar.addAction(self.actionNext_Image)
        self.toolBar.addAction(self.actionPrevious_Image)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Watching Prob Events", None))
        self.label.setText(_translate("MainWindow", "Filter probability", None))
        self.prob_lineEdit.setPlaceholderText(_translate("MainWindow", "prob number...", None))
        self.label_2.setText(_translate("MainWindow", "Select Color Map", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit", None))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar", None))
        self.actionLoad.setText(_translate("MainWindow", "Load file...", None))
        self.actionLoad.setShortcut(_translate("MainWindow", "Ctrl+L", None))
        self.actionEraser.setText(_translate("MainWindow", "Eraser", None))
        self.actionEraser.setToolTip(_translate("MainWindow", "Erase events from image", None))
        self.actionNext_Image.setText(_translate("MainWindow", "Next Image", None))
        self.actionNext_Image.setShortcut(_translate("MainWindow", "Right", None))
        self.actionPrevious_Image.setText(_translate("MainWindow", "Previous Image", None))
        self.actionPrevious_Image.setShortcut(_translate("MainWindow", "Left", None))
        self.actionUndo.setText(_translate("MainWindow", "Undo", None))
        self.actionUndo.setShortcut(_translate("MainWindow", "Ctrl+Z", None))

import resources_rc

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

