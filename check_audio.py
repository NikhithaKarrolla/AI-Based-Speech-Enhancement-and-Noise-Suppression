from src.audio.check_audio import check_audio_file


audio_path = (
    "data/raw/clean/clean_original.wav"
)


success = check_audio_file(
    audio_path
)


if success:

    print("\nAudio file is valid.")

else:

    print(
        "\nAudio file needs to be converted "
        "to a valid WAV/audio format."
    )