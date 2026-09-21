const audioFile = document.getElementById("audioFile");
const enhanceButton = document.getElementById("enhanceButton");
const statusText = document.getElementById("status");

const resultCard = document.getElementById("resultCard");
const audioPlayer = document.getElementById("audioPlayer");
const downloadButton = document.getElementById("downloadButton");


enhanceButton.addEventListener("click", async () => {

    if (!audioFile.files.length) {
        statusText.textContent = "Please select an audio file.";
        return;
    }

    const file = audioFile.files[0];

    const formData = new FormData();

    formData.append("file", file);

    statusText.textContent = "Enhancing audio...";
    enhanceButton.disabled = true;

    try {

        const response = await fetch(
            "/enhance",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error("Enhancement failed");
        }

        statusText.textContent =
            "Speech enhancement completed.";

        audioPlayer.src = data.output_file;

        downloadButton.href =
            data.output_file;

        resultCard.style.display = "block";

    } catch (error) {

        console.error(error);

        statusText.textContent =
            "Error while enhancing audio.";

    } finally {

        enhanceButton.disabled = false;

    }

});