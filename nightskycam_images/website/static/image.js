// Image detail page: stretch toggle (swaps the JPEG conversion URL),
// loading spinner while a conversion is in flight, and keyboard prev/next.
(function () {
    "use strict";

    const image = document.getElementById("main-image");
    const spinner = document.getElementById("image-spinner");
    const downloadJpeg = document.getElementById("download-jpeg");

    function setStretch(value) {
        const url = image.dataset.jpegUrl + "?stretch=" + value;
        if (spinner) spinner.style.display = "block";
        image.style.opacity = "0.4";
        image.src = url;
        if (downloadJpeg) downloadJpeg.href = url;
    }

    if (image) {
        image.addEventListener("load", function () {
            if (spinner) spinner.style.display = "none";
            image.style.opacity = "1";
        });
        image.addEventListener("error", function () {
            if (spinner) spinner.style.display = "none";
            image.style.opacity = "1";
        });
        // The first conversion of a big TIFF can take a while: show the
        // spinner until the initial load completes too.
        if (!image.complete && spinner) spinner.style.display = "block";
    }

    document.querySelectorAll('input[name="stretch"]').forEach(function (radio) {
        radio.addEventListener("change", function () {
            if (radio.checked) setStretch(radio.value);
        });
    });

    document.addEventListener("keydown", function (event) {
        if (event.key === "ArrowLeft") {
            const prev = document.getElementById("nav-prev");
            if (prev && !prev.classList.contains("disabled")) prev.click();
        } else if (event.key === "ArrowRight") {
            const next = document.getElementById("nav-next");
            if (next && !next.classList.contains("disabled")) next.click();
        }
    });
})();
