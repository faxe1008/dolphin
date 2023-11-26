import argparse
import glob
from constants import *
import librosa
import time
import sys
import pyaudio
import numpy as np
import librosa
from tensorflow import keras
from extract import extract_features
import os
import matplotlib.pyplot as plt


class EvaluationResult:
    def __init__(self, total_time, confidences):
        self.total_time = total_time
        self.confidences = confidences


class TimeRange:
    def __init__(self, start_time, end_time):
        start_segments = start_time.strip().split(":")
        end_segments = end_time.strip().split(":")

        assert len(start_segments) == 2
        assert len(end_segments) == 2

        self.start_seconds = int(start_segments[0]) * 60 + int(start_segments[1])
        self.end_seconds = int(end_segments[0]) * 60 + int(end_segments[1])


def run_model(model_path, audio_data, sample_rate, confidence_limit):
    model = keras.models.load_model(model_path)
    len_per_sample = sample_rate * SAMPLE_LENGTH_SEC

    showering_confidences = []
    total_showering_time = 0

    for i in range(0, len(audio_data), len_per_sample):
        sample = audio_data[i : i + len_per_sample]

        if len(sample) != len_per_sample:
            continue

        features = extract_features(sample, sample_rate).reshape(
            -1, N_MELS, FEATURE_COUNT, 1
        )
        network_output = model.predict(
            features, verbose=0, use_multiprocessing=True, batch_size=150
        )
        class_confidences = np.mean(network_output, axis=0)
        showering_confidences.append(class_confidences[0])

        duration = SAMPLE_LENGTH_SEC
        if class_confidences[0] > confidence_limit:
            total_showering_time = total_showering_time + SAMPLE_LENGTH_SEC

    return EvaluationResult(total_showering_time, showering_confidences)


def calculate_real_duration(time_ranges):
    duration = 0
    for trange in time_ranges:
        duration = duration + (trange.end_seconds - trange.start_seconds)
    return duration


def plot_evaluation_results(
    eval_results, time_ranges, confidence_limit, binarize=False
):
    if len(eval_results) == 0:
        return

    x = np.arange(
        0,
        len(eval_results[list(eval_results.keys())[0]].confidences) * SAMPLE_LENGTH_SEC,
        SAMPLE_LENGTH_SEC,
    )
    fig, (conf_plot, time_plot) = plt.subplots(2)

    for key, result in eval_results.items():
        y = result.confidences
        if binarize:
            y = [1.0 if e >= confidence_limit else 0.0 for e in y]
        conf_plot.plot(x, y, label=f"{key}: {result.total_time}")

    # Confidence limit line
    conf_plot.axhline(y=confidence_limit, color="r", linestyle="-")

    # Time range of actual showering
    colors = ["g", "b", "y"]
    for count, time_range in enumerate(time_ranges):
        conf_plot.axvline(
            x=time_range.start_seconds,
            color=colors[count % len(colors)],
            linestyle="dashed",
        )
        conf_plot.axvline(
            x=time_range.end_seconds,
            color=colors[count % len(colors)],
            linestyle="dashed",
        )

    conf_plot.set_title("Confidence Test Audio")
    conf_plot.set_xlabel("time [s]")
    conf_plot.set_ylabel("confidence [%]")
    leg = conf_plot.legend(loc="upper right")

    lined = {}  # Will map legend lines to original lines.
    for legline, origline in zip(leg.get_lines(), conf_plot.get_lines()):
        legline.set_picker(True)  # Enable picking on the legend line.
        lined[legline] = origline
    
    def on_pick(event):
        # On the pick event, find the original line corresponding to the legend
        # proxy line, and toggle its visibility.
        legline = event.artist
        origline = lined[legline]
        visible = not origline.get_visible()
        origline.set_visible(visible)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled.
        legline.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', on_pick)

    # time plot
    names = []
    durations = []
    for key, result in eval_results.items():
        durations.append(result.total_time)
        names.append(key)
    time_plot.bar(names, durations)

    if len(time_ranges) > 0:
        time_plot.axhline(
            y=calculate_real_duration(time_ranges), color="r", linestyle="-"
        )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Validator for dolphin model",
        description="Pits different models against a sample and checks the accuracy",
    )
    parser.add_argument(
        "-t",
        "--timestamps",
        type=str,
        help='Timestamp of the showering segments in the format of "00:00-01:20,01:30-01:54"',
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=float,
        default=0.5,
        help="Minimum confidence value to count as showering",
    )
    parser.add_argument("-b", "--binarize", action="store_true")
    parser.add_argument("songfile")
    parser.add_argument(
        "model",
        nargs="+",
        type=str,
        help="Path to a single model to evaluate accurancy for",
    )
    args = parser.parse_args()

    if len(args.model) == 0:
        print("Please provide a model")
        sys.exit(1)

    time_ranges = []
    if args.timestamps:
        for segment in args.timestamps.strip().split(","):
            times = segment.strip().split("-")
            time_ranges.append(TimeRange(times[0], times[1]))

    audio_data, sample_rate = librosa.load(args.songfile, sr=RATE)
    evaluation_results = dict()

    for model in args.model:
        evaluation_results[os.path.basename(model)] = run_model(
            model, audio_data, sample_rate, args.limit
        )

    plot_evaluation_results(evaluation_results, time_ranges, args.limit, args.binarize)
