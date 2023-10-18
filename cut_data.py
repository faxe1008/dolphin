import sys
import json
import os


class AudioSegment:
    def __init__(self, file, label, start, end):
        self.file = file
        self.label = label
        self.start = start
        self.end = end

    def __str__(self):
        return f"{self.label} [{self.start}-{self.end}]"


def parse_audio_segments(label_file):
    segments_per_file = {}

    with open(label_file, "r", encoding="utf-8") as lfile:
        label_json = json.load(lfile)

        known_labels = label_json["labels"]
        assert len(known_labels) > 0

        data = label_json["data"]
        assert len(data) > 0

        for file in data:
            segments_per_file[file] = []

            file_obj = data[file]
            for segment in file_obj:
                segment_label = segment["label"]
                segment_start = segment["start"]
                segment_end = segment["end"]

                segments_per_file[file].append(
                    AudioSegment(file, segment_label, segment_start, segment_end)
                )

    return segments_per_file, known_labels


def folder_from_label(label):
    return f"./data/{label}"


def create_data_folders(known_labels):
    for label in known_labels:
        os.makedirs(folder_from_label(label), exist_ok=True)


def time_to_seconds(time):
    parts = time.split(":")
    assert len(parts) == 2
    return int(parts[0]) * 60 + int(parts[1])


def cut_single_file(filename, dest, segments):
    if len(segments) == 0:
        return
    cmd_str = f"ffmpeg -y -i {filename} -af \"aselect='"
    cmd_betweens = []
    for segment in segments:
        cmd_betweens.append(
            "between(t,"
            + str(time_to_seconds(segment.start))
            + ","
            + str(time_to_seconds(segment.end))
            + ")"
        )

    cmd_str += "+".join(cmd_betweens)
    cmd_str += f"', asetpts=N/SR/TB\" {dest}/{os.path.splitext(os.path.basename(filename))[0]}_segments.wav"

    os.system(cmd_str)


def cut_into_folders(audio_segments_per_file, known_labels):
    for file in audio_segments_per_file:
        for label in known_labels:
            segment_list = list(
                filter(
                    lambda segment: segment.label == label,
                    audio_segments_per_file[file],
                )
            )
            cut_single_file(
                f"./data/raw/{file}", folder_from_label(label), segment_list
            )


def combine_labeled_audio(known_labels):
    for label in known_labels:
        folder_name = folder_from_label(label)
        cmd_str = f"find {folder_name}/*.wav | sed 's:\ :\\\ :g'| sed 's/^/file /' > fl.txt; ffmpeg -y -f concat -safe 0 -i fl.txt -c copy ./data/{label}.wav; rm fl.txt"
        os.system(cmd_str)


if __name__ == "__main__":
    audio_segments, known_labels = parse_audio_segments(sys.argv[1])
    create_data_folders(known_labels)
    cut_into_folders(audio_segments, known_labels)
    combine_labeled_audio(known_labels)
