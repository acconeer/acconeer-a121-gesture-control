import numpy as np
from enum import Enum
from acconeer.exptool.a121._core.peripherals import load_record

from acconeer.exptool.a121.algo._utils import get_approx_fft_vels
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import pickle

class gesture_detection_method(Enum):
    filter = 1
    ml = 2


def save_pickle(path, save_dict):
    with open(path, "wb") as f:
        pickle.dump(save_dict, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def eval_gesture_performed(feature_fifo, time_steps, nbr_fft_bins_to_evaluate, threshold):
    gesture_metric = np.std(np.roll(feature_fifo,nbr_fft_bins_to_evaluate,axis=1)[:time_steps, :nbr_fft_bins_to_evaluate*2])
    return (threshold < gesture_metric), gesture_metric


def calc_features(data, w):
    data -= np.mean(data, axis=0)
    feature = np.fft.fftshift(np.fft.fft(data*w, axis=0), axes=0)
    feature = np.abs(feature)
    feature = np.max(feature, axis=1)
    return feature


def extract_features(filepaths):
    features = []
    data_labels = []

    print(filepaths)

    for _, filepath in enumerate(filepaths):
        print("Loading...", filepath)

        # load data
        record = load_record(filepath)
        data_frames = record.frames
        data_frames = data_frames.squeeze()
        (nbr_frames, _, _) = data_frames.shape

        # extract sensor configuration
        sensor_config = record.session_config.sensor_config
        w = np.hanning(sensor_config.sweeps_per_frame)[:, None]
        w /= np.sum(w)

        # extract label information from filename
        label_idx_start = filepath.find("label_") + 6
        label_idx_stop = filepath.find("_idx_")
        label = filepath[label_idx_start:label_idx_stop]

        # loop over frames and extract features
        for frame_idx in range(nbr_frames):
            data = data_frames[frame_idx]
            feature = calc_features(data, w)
            features.append(feature)
            data_labels.append(label)

    vels, _ = get_approx_fft_vels(sensor_config)
    return features, data_labels, vels


def get_processing_config():

    return {
            "gd_num_samples_thres": 5,
            "gd_window_start_margin": 5,
            "gd_window_length": 10,
            "gd_num_bins": 4,
            "gd_method": gesture_detection_method.filter,
            "gd_gestures": ['dynamic_hand', 'static_hand'],
            "gc_window_length": 50,
            "gc_gestures": ['double_tap_in', 'tap_in', 'tap_out', 'wiggle'],
            }


def plot_training_result(history, model, X_test, y_test, gestures):
    fig,axs = plt.subplots(nrows=2)
    fig.set_figheight(8)
    fig.set_figwidth(16)
    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[1].plot(history.history['accuracy'])
    axs[1].plot(history.history['val_accuracy'])
    axs[1].set_xlabel('Epoch')
    axs[0].set_xlabel('Loss')
    axs[1].set_xlabel('Accuracy')
    axs[0].set_yscale('log')
    axs[0].grid()
    axs[1].grid()
    axs[0].legend(['Loss function', 'Validation data - Loss function'])
    axs[1].legend(['Prediction accuracy', 'Validation data - Prediction accuracy'])
    plt.show()

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gestures)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


def plot_features(gesture, features, vels):
    fig = plt.figure(figsize=(20,6))
    fig.suptitle('Gesture: ' + gesture, fontsize=24)
    plt.imshow(features, aspect='auto')
    plt.xlabel('Frame number', fontsize=24)
    plt.ylabel('Velocity (m/s)', fontsize=24)
    plt.yticks(ticks=range(len(vels)), labels=vels, fontsize=24)
    plt.xticks(fontsize=24)
    plt.show()
