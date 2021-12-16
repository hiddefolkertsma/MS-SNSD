"""
@author: chkarada
"""
import glob
import os
import json
import argparse
import configparser as cp
from tqdm import tqdm
import numpy as np
import pandas as pd
from audiolib import audioread, audiowrite, snr_mixer


def main(cfg):
    # Load parameters from config file
    snr_lower = float(cfg["snr_lower"])
    snr_upper = float(cfg["snr_upper"])
    total_snrlevels = float(cfg["total_snrlevels"])

    # Check if input directories exist
    clean_dir = os.path.join(os.path.dirname(__file__), "clean_train")
    if cfg["speech_dir"] != "None":
        clean_dir = cfg["speech_dir"]
    if not os.path.exists(clean_dir):
        assert False, "Clean speech data is required"

    noise_dir = os.path.join(os.path.dirname(__file__), "noise_train")
    if cfg["noise_dir"] != "None":
        noise_dir = cfg["noise_dir"]
    if not os.path.exists(noise_dir):
        assert False, "Noise data is required"

    # Load parameters from config file
    fs = float(cfg["sampling_rate"])
    audioformat = cfg["audioformat"]
    total_hours = float(cfg["total_hours"])
    audio_length = float(cfg["audio_length"])
    silence_lower = float(cfg["silence_lower"])
    silence_upper = float(cfg["silence_upper"])

    # Check if output directories exist
    output_dir = os.path.join(os.path.dirname(__file__), cfg["output_dir"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    noisyspeech_dir = os.path.join(output_dir, "noisy_speech")
    if not os.path.exists(noisyspeech_dir):
        os.makedirs(noisyspeech_dir)

    clean_proc_dir = os.path.join(output_dir, "clean_speech")
    if not os.path.exists(clean_proc_dir):
        os.makedirs(clean_proc_dir)

    noise_proc_dir = os.path.join(output_dir, "noise")
    if not os.path.exists(noise_proc_dir):
        os.makedirs(noise_proc_dir)

    vad_label_dir = os.path.join(output_dir, "vad_labels")
    if not os.path.exists(vad_label_dir):
        os.makedirs(vad_label_dir)

    total_secs = total_hours * 60 * 60
    total_samples = int(total_secs * fs)
    audio_length = int(audio_length * fs)
    snr = np.linspace(snr_lower, snr_upper, int(total_snrlevels))

    # Get filenames of the clean speech and the noise,
    # exclude anything that's set in `noisy_types_excluded`
    cleanfilenames = glob.glob(os.path.join(clean_dir, audioformat))
    if cfg["noise_types_excluded"] == "None":
        noisefilenames = glob.glob(os.path.join(noise_dir, audioformat))
    else:
        filestoexclude = cfg["noise_types_excluded"].split(",")
        noisefilenames = glob.glob(os.path.join(noise_dir, audioformat))
        for i in range(len(filestoexclude)):
            noisefilenames = [
                fn for fn in noisefilenames
                if not os.path.basename(fn).startswith(filestoexclude[i])
            ]

    filecounter = 0
    num_samples = 0

    # Main loop where the noise is being added to the speech
    while num_samples < total_samples:
        with tqdm(
                desc=
                f"({num_samples / fs / 3600:.2f}/{total_samples / fs / 3600:.2f} hours)"
        ) as pbar:
            # Pick a random noise file
            idx_n = np.random.randint(0, np.size(noisefilenames))
            noise, fs = audioread(noisefilenames[idx_n])

            # Trim noise if the noise is longer than the audio length
            if (len(noise) >= audio_length):
                noise = noise[0:audio_length]
            else:  # Add another noise file if the result is not long enough
                while len(noise) <= audio_length:
                    idx_n = idx_n + 1
                    if idx_n >= np.size(noisefilenames) - 1:
                        idx_n = np.random.randint(0, np.size(noisefilenames))
                    newnoise, fs = audioread(noisefilenames[idx_n])
                    noise = np.append(noise, newnoise)

            # Start speech somewhere in the first half of the audio clip
            start = np.random.randint(0, len(noise) / 2)

            # Pick a random speech file
            idx_s = np.random.randint(0, np.size(cleanfilenames))
            clean, fs = audioread(cleanfilenames[idx_s])

            labels = [(start, start + len(clean))]
            clean = np.append(np.zeros(start), clean)

            # Trim clean speech if the speech is longer than the noise length
            if (len(clean) > audio_length):
                clean = clean[0:audio_length]
                labels = [(start, audio_length)]
            else:  # Add another speech file if the result is not long enough
                while len(clean) < audio_length:
                    idx_s = idx_s + 1
                    if idx_s >= np.size(cleanfilenames) - 1:
                        idx_s = np.random.randint(0, np.size(cleanfilenames))
                    newclean, fs = audioread(cleanfilenames[idx_s])

                    # Random silence duration before next clip
                    silence_length = int(
                        fs * np.random.uniform(silence_lower, silence_upper))

                    # Start time of next audio clip
                    start = labels[-1][1] + silence_length

                    labels.append((start, start + len(newclean)))
                    cleanconcat = np.append(clean, np.zeros(silence_length))
                    clean = np.append(cleanconcat, newclean)

            clean = clean[0:len(noise)]

            assert len(clean) == len(noise)

            # Mix the clean speech and the noise together and write to output
            for i in range(np.size(snr)):
                clean_snr, noise_snr, noisy_snr = snr_mixer(clean=clean,
                                                            noise=noise,
                                                            snr=snr[i])
                noisyfilename = ("noisy" + str(filecounter) + "_SNRdb_" +
                                 str(snr[i]) + "_clnsp" + str(filecounter) +
                                 ".wav")
                cleanfilename = "clnsp" + str(filecounter) + ".wav"
                noisefilename = ("noise" + str(filecounter) + "_SNRdb_" +
                                 str(snr[i]) + ".wav")
                vadlabelfilename = "vad" + str(filecounter) + ".json"
                noisypath = os.path.join(noisyspeech_dir, noisyfilename)
                cleanpath = os.path.join(clean_proc_dir, cleanfilename)
                noisepath = os.path.join(noise_proc_dir, noisefilename)
                vadlabelpath = os.path.join(vad_label_dir, vadlabelfilename)
                audiowrite(noisy_snr, fs, noisypath, norm=False)
                audiowrite(clean_snr, fs, cleanpath, norm=False)
                audiowrite(noise_snr, fs, noisepath, norm=False)
                with open(vadlabelpath, "w") as vadlabelfile:
                    json.dump(labels, vadlabelfile)
                num_samples = num_samples + len(noisy_snr)

            filecounter = filecounter + 1
            pbar.update(1)

    # Create metadata file
    metadata = {'length': filecounter, 'snr': list(snr), 'fs': fs}
    with open(os.path.join(output_dir, "metadata.json"), 'w') as metadata_file:
        json.dump(metadata, metadata_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Configurations: read noisyspeech_synthesizer.cfg
    parser.add_argument(
        "--cfg",
        default="noisyspeech_synthesizer.cfg",
        help="Read noisyspeech_synthesizer.cfg for all the details",
    )
    parser.add_argument("--cfg_str", type=str, default="noisy_speech_train")
    args = parser.parse_args()

    cfgpath = os.path.join(os.path.dirname(__file__), args.cfg)
    assert os.path.exists(cfgpath), f"No configuration file as [{cfgpath}]"
    cfg = cp.ConfigParser(interpolation=cp.ExtendedInterpolation())
    cfg.read(cfgpath)
    main(cfg[args.cfg_str])
