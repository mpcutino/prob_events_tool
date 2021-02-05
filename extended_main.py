from os import remove
from PyQt4 import QtGui, QtCore

import cv2
import numpy as np
import rosbag
from matplotlib import cm
from cv_bridge import CvBridge

from qt_generated.main import Ui_MainWindow, _fromUtf8
from utils.qt_utils import show_message
from utils.constants import PROB_EVENT_TOPIC, IMG_TOPIC, IMG_H, IMG_W, EVENTS_PER_IMG, EV2Maintain_PER_IMG
from utils.img_utils import draw_bBox_from_cluster, to3channels, bitwise_img, draw_bBox_from_clusters
from modes.cluster import get_clusters


# noinspection PyAttributeOutsideInit
class Modified_MainWindow(Ui_MainWindow):

    def __init__(self):
        super(Modified_MainWindow, self).__init__()

        self.__init_items__()
        self.color_maps = ["hot", "viridis", "Greens", "copper"]
        self.modes = ["cluster", "prob_pixel", "binary"]
        self.group_by = ["pixels", "events"]
        self.prob_filter = 0
        self.minh = 0
        self.minw = 0
        self.class_index_selection = 0
        self.images_raw_messages = []
        self.img_msg_count = 0
        self.two_side_img_mode = False

    def doUiSetup(self, qtMainWindow):
        # this is from base class
        self.setupUi(qtMainWindow)

        # now add my own stuff
        self.mainWindow = qtMainWindow
        # add connections to actions
        self.actionLoad.triggered.connect(self.load_file)
        self.actionNext_Image.triggered.connect(self.next_image)
        self.actionPrevious_Image.triggered.connect(self.prev_image)
        self.actionMake_video.triggered.connect(self.make_video)
        # add items to color map comboBox
        self.cmComboBox.addItems(self.color_maps)
        self.cmComboBox.currentIndexChanged.connect(self.cbox_change)
        # add connection to prob filter change
        self.prob_lineEdit.returnPressed.connect(self.prob_filter_change)
        # add items to mode comboBox
        self.mode_comboBox.addItems(self.modes)
        self.mode_comboBox.setEnabled(False)
        self.mode_comboBox.currentIndexChanged.connect(self.mode_update)
        # set eps and min_point defaults
        self.eps_spinBox.setValue(15)
        self.eps_spinBox.setMinimum(1)
        self.minPoints_spinBox.setValue(50)
        self.minPoints_spinBox.setMinimum(2)
        self.eps_spinBox.valueChanged.connect(self.dummy_img_update)
        self.minPoints_spinBox.valueChanged.connect(self.dummy_img_update)
        # add items to dbscan group by
        self.dbscanGroupBy_comboBox.addItems(self.group_by)
        self.dbscanGroupBy_comboBox.currentIndexChanged.connect(self.dummy_img_update)
        self.dbscanGroupBy_comboBox.setEnabled(False)
        # add connections to min width and height
        self.minH_lineEdit.returnPressed.connect(self.minH_change)
        self.minW_lineEdit.returnPressed.connect(self.minW_change)
        self.minH_lineEdit.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.minW_lineEdit.setFocusPolicy(QtCore.Qt.ClickFocus)
        # add connection to clusterProb radio button
        self.clusterProb_RadioBtn.clicked.connect(self.dummy_img_update)
        # add connection to flip image radio button
        self.checkB_Reverse.clicked.connect(self.dummy_img_update)
        # add conection to class index combo box
        self.cbox_ClassIndex.addItems(["0", "1"])
        self.cbox_ClassIndex.currentIndexChanged.connect(self.class_index_change)

    def class_index_change(self):
        self.class_index_selection = int(self.cbox_ClassIndex.currentText())
        self.dummy_img_update()

    def make_video(self):
        if len(self.images):
            show_message("Video will be made with the images you have already seen", title="Warning!!")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter('output.avi', fourcc, 15.0, (IMG_W, IMG_H))

            for img in self.images:
                show_img = self.get_img_base_on_mode(img)
                color_map = self.cmComboBox.currentText()
                show_img = cm.get_cmap(str(color_map), 10)(show_img) * 255
                cv2.imwrite("tmp.png", show_img)
                r_img = cv2.imread("tmp.png")
                out.write(r_img)

            cv2.destroyAllWindows()
            out.release()

    def mode_update(self):
        if str(self.mode_comboBox.currentText()) == "cluster":
            self.dbscanGroupBy_comboBox.setEnabled(True)
            self.eps_spinBox.setEnabled(True)
            self.minPoints_spinBox.setEnabled(True)
            self.clusterProb_RadioBtn.setEnabled(True)
        else:
            self.dbscanGroupBy_comboBox.setEnabled(False)
            self.eps_spinBox.setEnabled(False)
            self.minPoints_spinBox.setEnabled(False)
            self.clusterProb_RadioBtn.setEnabled(False)
        self.dummy_img_update()

    def minH_change(self):
        h_value = self.some_lineEdit_change(self.minH_lineEdit)
        if h_value != self.minh:
            self.minh = h_value
            self.dummy_img_update()

    def minW_change(self):
        w_value = self.some_lineEdit_change(self.minW_lineEdit)
        if w_value != self.minw:
            self.minw = w_value
            self.dummy_img_update()

    def dummy_img_update(self):
        self.count -= 1
        self.update_image(forward=True)

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
            self.images_raw_messages = list(bag.read_messages(IMG_TOPIC))
            if len(self.images_raw_messages) and len(bag_messages):
                self.two_side_img_mode = True
            elif not len(self.images_raw_messages):
                self.two_side_img_mode = False
                self.img_msg_count = 0
            if len(bag_messages):
                print(len(bag_messages))
                self.__init_items__()
                self.bag_messages = bag_messages
                self.update_image(forward=True)
                self.eps_spinBox.setEnabled(True)
                self.minPoints_spinBox.setEnabled(True)
                self.mode_comboBox.setEnabled(True)
                self.dbscanGroupBy_comboBox.setEnabled(True)
                self.clusterProb_RadioBtn.setEnabled(True)
            else:
                show_message("Not a valid bag. It must contains the topic {0}".format(PROB_EVENT_TOPIC))

    def cbox_change(self, e):
        # trick to update the image
        self.count -= 1
        self.update_image(forward=True)

    def prob_filter_change(self):
        p_value = self.some_lineEdit_change(self.prob_lineEdit)
        if p_value != self.prob_filter:
            self.prob_filter = p_value
            # update the image
            self.count -= 1
            self.update_image(forward=True)

    @staticmethod
    def some_lineEdit_change(lineEdit):
        line_edit_text = lineEdit.text()
        lineEdit.clearFocus()
        try:
            p_filter = int(str(line_edit_text))
            return p_filter
        except Exception as e:
            print e
            if len(str(line_edit_text)):
                show_message("Not a number. Using 0 as default.")
                lineEdit.setText("")
        return 0

    def next_image(self):
        self.update_image(forward=True)

    def prev_image(self):
        self.update_image(forward=False)

    def update_image(self, forward=True):
        if len(self.bag_messages):
            self.count += 1 if forward else -1

            if self.count < 0 and not forward:
                self.count = len(self.images) - 1 if self.has_finish_bag else 0
            if self.count >= len(self.images) and forward:
                if self.has_finish_bag:
                    self.count = 0
                else:
                    # compute the new latest events and store it in images array
                    new_ev = self.get_new_latest_events()
                    self.has_finish_bag = self.bag_msg_count == len(self.bag_messages)
                    self.images.append(new_ev)
            show_img = self.get_img_base_on_mode(self.images[self.count])
            if self.checkB_Reverse.isChecked():
                show_img = np.rot90(show_img, 2)
            color_map = self.cmComboBox.currentText()
            # show_img = cm.get_cmap(str(color_map), 10)(show_img)*255
            cv2.imwrite("tmp.png", show_img)
            show_img = cv2.imread("tmp.png")
            self.img_lbl.setPixmap(QtGui.QPixmap("tmp.png"))
            if len(self.images[self.count]):
                self.lbl_ImgName.setText(str(self.images[self.count][-1].ts))

            if self.two_side_img_mode and self.count < len(self.images_raw_messages):
                # show traditional image
                bridge = CvBridge()
                msg = self.images_raw_messages[self.count]
                img = bridge.imgmsg_to_cv2(msg.message, desired_encoding="passthrough")
                img = to3channels(img, show_img)
                img = bitwise_img(img, show_img)
                cv2.imwrite("tmp.png", img)
                self.img_lbl_rgb.setPixmap(QtGui.QPixmap("tmp.png"))
            remove("tmp.png")

    def get_img_base_on_mode(self, events):
        """
        The idea is to use different modes in the showing method, and we need the events for that.
        #   mode 1) showing the probability colored image using only the value of the event
        #   mode 2) showing the binary image without talking into account the probability
        #   mode 3) cluster events and build the detector over the clusters
        :param events:
        :return:
        """
        if self.mode_comboBox.currentText() == "cluster":
            eps = self.eps_spinBox.value()
            min_samples = self.minPoints_spinBox.value()
            group_by_pixels = str(self.dbscanGroupBy_comboBox.currentText()) == "pixels"
            use_cluster_prob = self.clusterProb_RadioBtn.isChecked()
            clusters, ev_of_interest = get_clusters(events, self.class_index_selection, eps=eps,
                                                    min_samples=min_samples, use_unique_events=group_by_pixels)
            # new_image = draw_bBox_from_cluster(clusters, ev_of_interest, events,
            #                                    self.class_index_selection, prob_filter=self.prob_filter / 100.0,
            #                                    min_dims=(self.minh, self.minw), use_cluster_prob=use_cluster_prob)
            new_image = draw_bBox_from_clusters(clusters, ev_of_interest, events,
                                                self.class_index_selection, prob_filter=self.prob_filter / 100.0,
                                                min_dims=(self.minh, self.minw), use_cluster_prob=use_cluster_prob)
        else:
            new_image = np.zeros((IMG_H, IMG_W))
            binary_mode = self.mode_comboBox.currentText() == "binary"
            for ev in events:
                value = self.get_binary_prob_value(ev) if binary_mode else self.get_colored_prob_value(ev)
                new_image[ev.y, ev.x] = value
            # zeroing values below the prob filter
            new_image[new_image < self.prob_filter / 100.0] = 0
        return new_image

    def get_new_latest_events(self):
        latest_events = []
        if self.two_side_img_mode:
            for msg in self.bag_messages[self.bag_msg_count:]:
                if self.img_msg_count >= len(self.images_raw_messages) \
                        or \
                        msg.timestamp < self.images_raw_messages[self.img_msg_count].timestamp:
                    self.bag_msg_count += 1
                    for e in msg.message.events:
                        latest_events.append(e)
                else:
                    # all events between the current image and the next has been analyzed
                    self.img_msg_count += 1
                    break
        else:
            latest_events = list(self.images[-1]) if len(self.images) else []
            if len(latest_events):
                latest_events = latest_events[len(latest_events) - EV2Maintain_PER_IMG:]
            ev_count = 0
            for msg in self.bag_messages[self.bag_msg_count:]:
                if ev_count > EVENTS_PER_IMG:
                    break
                self.bag_msg_count += 1
                for e in msg.message.events:
                    ev_count += 1
                    latest_events.append(e)
                # testing the use of just one message stamp
                # break
        return latest_events

    def get_binary_prob_value(self, e):
        # persons are index 0 in prob array
        if e.probs[self.class_index_selection] == max(e.probs):
            return 1
        return 0

    def get_colored_prob_value(self, e):
        # persons are index 0 in prob array
        return e.probs[self.class_index_selection]/100.0

    def __init_items__(self):
        self.bag_messages = []
        self.images = []
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
