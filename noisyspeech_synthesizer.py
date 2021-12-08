"""
@author: chkarada
"""
import glob
import numpy as np
import soundfile as sf
import os
import argparse
import configparser as CP
from audiolib import audioread, audiowrite, snr_mixer


def main(cfg):
    # Load parameters from config file
    snr_lower = float(cfg["snr_lower"])
    snr_upper = float(cfg["snr_upper"])
    total_snrlevels = float(cfg["total_snrlevels"])

    # Check if input directories exist
    clean_dir = os.path.join(os.path.dirname(__file__), 'clean_train')
    if cfg["speech_dir"] != 'None':
        clean_dir = cfg["speech_dir"]
    if not os.path.exists(clean_dir):
        assert False, ("Clean speech data is required")

    noise_dir = os.path.join(os.path.dirname(__file__), 'noise_train')
    if cfg["noise_dir"] != 'None':
        noise_dir = cfg["noise_dir"]
    if not os.path.exists(noise_dir):
        assert False, ("Noise data is required")

    # Load parameters from config file
    fs = float(cfg["sampling_rate"])
    audioformat = cfg["audioformat"]
    total_hours = float(cfg["total_hours"])
    audio_length = float(cfg["audio_length"])
    silence_length = float(cfg["silence_length"])

    # Check if output directories exist
    noisyspeech_dir = os.path.join(
        os.path.dirname(__file__), 'NoisySpeech_training')
    if not os.path.exists(noisyspeech_dir):
        os.makedirs(noisyspeech_dir)
    clean_proc_dir = os.path.join(
        os.path.dirname(__file__), 'CleanSpeech_training')
    if not os.path.exists(clean_proc_dir):
        os.makedirs(clean_proc_dir)
    noise_proc_dir = os.path.join(os.path.dirname(__file__), 'Noise_training')
    if not os.path.exists(noise_proc_dir):
        os.makedirs(noise_proc_dir)

    total_secs = total_hours*60*60
    total_samples = int(total_secs * fs)
    audio_length = int(audio_length*fs)
    SNR = np.linspace(snr_lower, snr_upper, total_snrlevels)

    # Get filenames of the clean speech and the noise, exclude anything that's set in `noisy_types_excluded`
    cleanfilenames = glob.glob(os.path.join(clean_dir, audioformat))
    if cfg["noise_types_excluded"] == 'None':
        noisefilenames = glob.glob(os.path.join(noise_dir, audioformat))
    else:
        filestoexclude = cfg["noise_types_excluded"].split(',')
        noisefilenames = glob.glob(os.path.join(noise_dir, audioformat))
        for i in range(len(filestoexclude)):
            noisefilenames = [fn for fn in noisefilenames if not os.path.basename(
                fn).startswith(filestoexclude[i])]

    filecounter = 0
    num_samples = 0

    # Main loop where the noise is being added to the speech
    while num_samples < total_samples:
        # Pick a random speech file
        idx_s = np.random.randint(0, np.size(cleanfilenames))
        clean, fs = audioread(cleanfilenames[idx_s])

        if len(clean) > audio_length:
            clean = clean  # ???

        else:

            # Add another speech file if the result is not long enough
            while len(clean) <= audio_length:
                idx_s = idx_s + 1
                if idx_s >= np.size(cleanfilenames)-1:
                    idx_s = np.random.randint(0, np.size(cleanfilenames))
                newclean, fs = audioread(cleanfilenames[idx_s])
                cleanconcat = np.append(
                    clean, np.zeros(int(fs*silence_length)))
                clean = np.append(cleanconcat, newclean)

        # Pick a random noise file
        idx_n = np.random.randint(0, np.size(noisefilenames))
        noise, fs = audioread(noisefilenames[idx_n])

        if len(noise) >= len(clean):  # Trim noise if the noise is longer than the clean audio
            noise = noise[0:len(clean)]

        else:

            # Add another noise file if the result is not long enough
            while len(noise) <= len(clean):
                idx_n = idx_n + 1
                if idx_n >= np.size(noisefilenames)-1:
                    idx_n = np.random.randint(0, np.size(noisefilenames))
                newnoise, fs = audioread(noisefilenames[idx_n])
                noiseconcat = np.append(
                    noise, np.zeros(int(fs*silence_length)))
                noise = np.append(noiseconcat, newnoise)
        noise = noise[0:len(clean)]
        filecounter = filecounter + 1

       # Mix the clean speech and the noise together and write to output
        for i in range(np.size(SNR)):
            clean_snr, noise_snr, noisy_snr = snr_mixer(
                clean=clean, noise=noise, snr=SNR[i])
            noisyfilename = 'noisy' + \
                str(filecounter)+'_SNRdb_' + \
                str(SNR[i])+'_clnsp'+str(filecounter)+'.wav'
            cleanfilename = 'clnsp'+str(filecounter)+'.wav'
            noisefilename = 'noisy' + \
                str(filecounter)+'_SNRdb_'+str(SNR[i])+'.wav'
            noisypath = os.path.join(noisyspeech_dir, noisyfilename)
            cleanpath = os.path.join(clean_proc_dir, cleanfilename)
            noisepath = os.path.join(noise_proc_dir, noisefilename)
            audiowrite(noisy_snr, fs, noisypath, norm=False)
            audiowrite(clean_snr, fs, cleanpath, norm=False)
            audiowrite(noise_snr, fs, noisepath, norm=False)
            num_samples = num_samples + len(noisy_snr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Configurations: read noisyspeech_synthesizer.cfg
    parser.add_argument("--cfg", default="noisyspeech_synthesizer.cfg",
                        help="Read noisyspeech_synthesizer.cfg for all the details")
    parser.add_argument("--cfg_str", type=str, default="noisy_speech")
    args = parser.parse_args()

    cfgpath = os.path.join(os.path.dirname(__file__), args.cfg)
    assert os.path.exists(cfgpath), f"No configuration file as [{cfgpath}]"
    cfg = CP.ConfigParser()
    cfg._interpolation = CP.ExtendedInterpolation()
    cfg.read(cfgpath)

    main(cfg._sections[args.cfg_str])
