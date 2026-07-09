// Filter-form behavior: classifier sliders enable/disable + live value
// labels, and dropping empty fields from the submitted query string.
(function () {
    "use strict";

    document.querySelectorAll(".classifier-toggle").forEach(function (toggle) {
        const slider = document.getElementById(toggle.dataset.slider);
        if (!slider) return;
        toggle.addEventListener("change", function () {
            slider.disabled = !toggle.checked;
        });
    });

    document.querySelectorAll(".classifier-slider").forEach(function (slider) {
        const output = document.getElementById(slider.dataset.output);
        if (!output) return;
        slider.addEventListener("input", function () {
            output.textContent = Number(slider.value).toFixed(2);
        });
    });

    // Disable empty inputs on submit so the URL stays clean (disabled
    // fields are not serialized into the query string).
    const form = document.getElementById("filter-form");
    if (form) {
        form.addEventListener("submit", function () {
            form.querySelectorAll("input, select").forEach(function (el) {
                if (el.disabled) return;
                const empty =
                    (el.tagName === "SELECT" && el.multiple)
                        ? el.selectedOptions.length === 0
                        : el.value === "";
                if (empty) el.disabled = true;
            });
        });
    }
})();
