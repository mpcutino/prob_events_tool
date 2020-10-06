from os import remove
from PyQt4 import QtGui, QtCore

import cv2
import numpy as np
import rosbag
from matplotlib import cm

from qt_generated.main import Ui_MainWindow, _fromUtf8
from utils.qt_utils import show_message
from utils.constants import PROB_EVENT_TOPIC, IMG_H, IMG_W, EVENTS_PER_IMG, EV2Maintain_PER_IMG
from modes.cluster import get_cluster_image


# noinspection PyAttributeOutsideInit
class Modified_MainWindow(Ui_MainWindow):

    def __init__(self):
        super(Modified_MainWindow, self).__init__()

        self.__init_items__()
        self.color_maps = ["hot", "viridis", "Greens", "copper"]
        self.prob_filter = 0

    def doUiSetup(self, qtMainWindow):
        # this is from base class
        self.setupUi(qtMainWindow)

        # now add my own stuff
        self.mainWindow = qtMainWindow
        # add connections to actions
        self.actionLoad.triggered.connect(self.load_file)
        self.actionNext_Image.triggered.connect(self.next_image)
        self.actionPrevious_Image.triggered.connect(self.prev_image)
        # add items to comboBox
        self.cmComboBox.addItems(self.color_maps)
        self.cmComboBox.currentIndexChanged.connect(self.cbox_change)
        # add connection to prob filter change
        self.prob_lineEdit.returnPressed.connect(self.prob_filter_change)

    def load_file(self):
        # The QWidget widget is the base class of all user interface objects in PyQt4.
        w = QtGui.QWidget()
        # Set window size.
        w.resize(320, 240)
        # Set window title
        w.setWindowTitle("Hello World!")
        filename = QtGui.QFileDialog.getOpenFileName(w, 'Open File', '/', "Bag Files (*.bag)")
        print(filename)

        if len(str(filename)):
            bag = rosbag.Bag(str(filename))
            bag_messages = list(bag.read_messages(PROB_EVENT_TOPIC))
            if len(bag_messages):
                self.__init_items__()
                self.bag_messages = bag_messages
                self.update_image(forward=True)
            else:
                show_message("Not a valid bag. It must contains the topic {0}".format(PROB_EVENT_TOPIC))

    def cbox_change(self, e):
        # trick to update the image
        self.count -= 1
        self.update_image(forward=True)

    def prob_filter_change(self):
        line_edit_text = self.prob_lineEdit.text()
        if not len(str(line_edit_text)):
            if self.prob_filter > 0:
                self.prob_filter = 0
                # update the image
                self.count -= 1
                self.update_image(forward=True)
        else:
            try:
                p_filter = int(str(line_edit_text))
                self.prob_filter = p_filter
                # update the image
                self.count -= 1
                self.update_image(forward=True)
            except Exception as e:
                print e
                show_message("Probability filter must be a number. "
                             "It shows only events with higher probability than the filter")
                self.prob_lineEdit.setText("")
        self.prob_lineEdit.clearFocus()

    def next_image(self):
        self.update_image(forward=True)

    def prev_image(self):
        self.update_image(forward=False)

    def update_image(self, forward=True):
        self.count += 1 if forward else -1

        if self.count < 0 and not forward:
            self.count = len(self.images) - 1 if self.has_finish_bag else 0
        if self.count >= len(self.images) and forward:
            if self.has_finish_bag:
                self.count = 0
            else:
                # compute the new image and store it in images array
                new_image = self.get_new_img()
                self.has_finish_bag = self.bag_msg_count == len(self.bag_messages)
                self.images.append(new_image)
        color_map = self.cmComboBox.currentText()
        show_img = np.array(self.images[self.count])
        show_img[show_img < self.prob_filter/100.0] = 0
        show_img = cm.get_cmap(str(color_map), 10)(show_img)*255
        cv2.imwrite("tmp.png", show_img)
        self.img_lbl.setPixmap(QtGui.QPixmap("tmp.png"))
        remove("tmp.png")

    def get_new_img(self):
        # TODO
        #  In the loop update only the latest_events list. Then use this list to construct the image.
        #  Also, save this list instead of the numpy image.
        #  The idea is to use different modes in the showing method, and we need the events for that.
        #   mode 1) showing the probability colored image using only the value of the event
        #   mode 2) showing the binary image without talking into account the probability
        #   mode 3) cluster events and build the detector over the clusters
        #  Mode 3 needs to be done, and everything needs to be modified to admit different modes
        new_image = self.init_img_from_latest_events()
        self.latest_events = self.latest_events[len(self.latest_events) - EV2Maintain_PER_IMG:]
        ev_count = 0
        for msg in self.bag_messages[self.bag_msg_count:]:
            if ev_count > EVENTS_PER_IMG:
                break
            self.bag_msg_count += 1
            for e in msg.message.events:
                ev_count += 1
                # new_image[e.y, e.x] = self.get_binary_prob_value(e)
                new_image[e.y, e.x] = self.get_colored_prob_value(e)
        return new_image

    def get_binary_prob_value(self, e):
        # persons are index 0 in prob array
        if e.probs[0] == max(e.probs):
            self.latest_events.append(e)
            return 1
        return 0

    def get_colored_prob_value(self, e):
        # persons are index 0 in prob array
        if e.probs[0] == max(e.probs):
            self.latest_events.append(e)
        return e.probs[0]/100.0

    def init_img_from_latest_events(self):
        if len(self.images):
            # we need a copy to prevent the modification of the last image
            new_image = np.array(self.images[-1])
            for i, e in enumerate(self.latest_events):
                if i > len(self.latest_events) - EV2Maintain_PER_IMG: break
                new_image[e.y, e.x] = 0
        else:
            new_image = np.zeros((IMG_H, IMG_W))
        return new_image
        # return np.zeros((IMG_H, IMG_W))

    def __init_items__(self):
        self.bag_messages = []
        self.images = []
        self.latest_events = []
        self.count = -1
        self.bag_msg_count = 0
        self.has_finish_bag = False


if __name__ == "__main__":
    import sys

    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Modified_MainWindow()
    ui.doUiSetup(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
