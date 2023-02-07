import copy
import numpy as np
import absl.logging
import pyqtgraph as pg
import tensorflow as tf
from tensorflow import keras
absl.logging.set_verbosity(absl.logging.ERROR)

from acconeer.exptool.a121.algo._utils import get_distances_m, get_approx_fft_vels

import util
from acconeer.exptool.pg_process import PGProcess

from acconeer.exptool.a121._core.peripherals import load_record

import acconeer.exptool as et
from acconeer.exptool import a121
tf.compat.v1.disable_eager_execution()


def main():

    args = a121.ExampleArgumentParser().parse_args()
    et.utils.config_logging(args)

    client = a121.Client(**a121.get_client_args(args))
    client.connect()

    sensor_config = load_record('data/conf.h5').session_config.sensor_config
    metadata = client.setup_session(sensor_config)

    processor = Processor(sensor_config)
    pg_updater = PGUpdater(sensor_config, metadata)
    pg_process = PGProcess(pg_updater)
    pg_process.start()

    client.start_session()
    interrupt_handler = et.utils.ExampleInterruptHandler()

    while not interrupt_handler.got_signal:
        data = client.get_next()
        plot_data = processor.process(data)
        pg_process.put_data(plot_data)

    print("\nDisconnecting...")
    client.disconnect()


class Processor:
    def __init__(self, sensor_config):

        self.w = np.hanning(sensor_config.sweeps_per_frame)[:, None]
        self.w /= np.sum(self.w)

        processing_config = util.get_processing_config()

        self.gc_model = keras.models.load_model('gc_model')
        self.gd_model = keras.models.load_model('gd_model')

        self.gd_method = processing_config["gd_method"]
        self.gd_num_bins = processing_config["gd_num_bins"]
        self.gd_num_samples_thres = processing_config["gd_num_samples_thres"]
        self.gd_threshold = util.load_pickle("gd_threshold")["gd_threshold"]
        self.gd_window_length = processing_config["gd_window_length"]
        self.gd_window_start_margin = processing_config["gd_window_start_margin"]

        self.gc_gestures = processing_config["gc_gestures"]
        self.gc_num_featutes = self.gc_model.layers[0].output_shape[-1]
        self.gc_window_length = processing_config["gc_window_length"]

        self.feature_fifo = np.zeros(shape=(self.gc_window_length, self.gc_num_featutes))
        self.feature_fifo_evaluated = np.zeros(shape=(self.gc_window_length, self.gc_num_featutes))

        self.gd_gesture_found = False
        self.gd_gesture_started_counter = 0
        self.gc_gesture_length = 0

        self.predicted_class = ""

    def process(self, data):
        data = data.frame

        feature = util.calc_features(data, self.w)
        self.feature_fifo[1:,:] = self.feature_fifo[:-1,:]
        self.feature_fifo[0,:] = feature

        if self.gd_method == util.gesture_detection_method.filter:
            gesture_performed, _ = util.eval_gesture_performed(
                                                            self.feature_fifo,
                                                            self.gd_window_length,
                                                            self.gd_num_bins,
                                                            threshold=self.gd_threshold
                                                            )
        else:
            y_pred = self.gd_model.predict(np.expand_dims(np.expand_dims(self.feature_fifo[:self.gd_window_length,:], axis=-1),axis=0))[0]
            gesture_performed = True if 0.95 < y_pred[0] else False

        if gesture_performed and not self.gd_gesture_found:
            self.gd_gesture_started_counter += 1
            if self.gd_num_samples_thres <= self.gd_gesture_started_counter:
                self.gd_gesture_found = True
        else:
            self.gd_gesture_started_counter = 0

        process_fifo = False
        if self.gd_gesture_found and not gesture_performed:
            if (self.gc_window_length - (self.gd_num_samples_thres + self.gd_window_start_margin + self.gd_window_length)) <= self.gc_gesture_length:
                process_fifo = True

        if self.gd_gesture_found:
            self.gc_gesture_length += 1

        if process_fifo:
            self.feature_fifo_evaluated = copy.copy(np.flipud(self.feature_fifo))
            y_pred = self.gc_model.predict(np.expand_dims(self.feature_fifo_evaluated, axis=0))[0]
            self.predicted_class = self.gc_gestures[np.argmax(y_pred)]

            self.gc_gesture_length = 0
            self.gd_gesture_found = False
            self.gd_gesture_started_counter = 0

        return {"fifo": self.feature_fifo_evaluated, "fifo_cont": self.feature_fifo, "predicted_class": self.predicted_class}


class PGUpdater:
    def __init__(self, sensor_config, metadata):
        self.depths_m, self.step_length_m = get_distances_m(sensor_config, metadata)
        self.vels, self.vel_res = get_approx_fft_vels(metadata, sensor_config)

    def setup(self, win):
        tr = pg.QtGui.QTransform()
        tr.translate(self.depths_m[0], self.vels[0] - 0.5 * self.vel_res)
        tr.scale(1, self.vel_res)

        self.cont_plot = win.addPlot()
        self.cont_plot.setMenuEnabled(False)
        self.cont_plot.setLabel("bottom", "Time[Frame]")
        self.cont_plot.setLabel("left", "Velocity (m/s)")
        self.cont_im = pg.ImageItem(autoDownsample=True)
        self.cont_im.setLookupTable(et.utils.pg_mpl_cmap("viridis"))
        self.cont_plot.addItem(self.cont_im)
        self.cont_im.setTransform(tr)

        win.nextRow()

        self.ft_plot = win.addPlot()
        self.ft_plot.setMenuEnabled(False)
        self.ft_plot.setLabel("bottom", "Time[Frame]")
        self.ft_plot.setLabel("left", "Velocity (m/s)")
        self.ft_im = pg.ImageItem(autoDownsample=True)
        self.ft_im.setLookupTable(et.utils.pg_mpl_cmap("viridis"))
        self.ft_plot.addItem(self.ft_im)
        self.ft_im.setTransform(tr)

        self.html = (
            '<div style="text-align: center">'
            '<span style="color: #FFFFFF;font-size:20pt;">'
            "{}</span></div>"
        )
        self.text_item = pg.TextItem()
        self.text_item.setPos(0, self.vels[-1])
        self.ft_plot.addItem(self.text_item)
        self.text_item.show()


    def update(self, d):
        self.cont_im.updateImage(np.flipud(d["fifo_cont"]), levels=(0, 1.05 * np.max(d["fifo_cont"])))
        self.ft_im.updateImage(d["fifo"], levels=(0, 1.05 * np.max(d["fifo"])))
        if d["predicted_class"] == "":
            self.text_item.setHtml(self.html.format("Go-ahead and perform gestures"))
        else:
            self.text_item.setHtml(self.html.format(d["predicted_class"].replace("_"," ")))


if __name__ == "__main__":
    main()
