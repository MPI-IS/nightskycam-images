// Ask page: chat client for POST /api/ask. History lives in memory for
// the page lifetime; the full history is sent with every request
// (stateless server). Model text is rendered via textContent only.
(function () {
    "use strict";

    const log = document.getElementById("chat-log");
    const form = document.getElementById("ask-form");
    const input = document.getElementById("ask-input");
    const sendButton = document.getElementById("ask-send");
    const sendIcon = document.getElementById("ask-send-icon");
    const spinner = document.getElementById("ask-spinner");
    const resetButton = document.getElementById("ask-reset");

    let history = [];
    const HISTORY_LIMIT = 12;

    function bubble(role) {
        const el = document.createElement("div");
        el.className = "chat-bubble " + role;
        log.appendChild(el);
        el.scrollIntoView({behavior: "smooth", block: "end"});
        return el;
    }

    function renderText(el, text) {
        const p = document.createElement("div");
        p.className = "chat-text";
        p.textContent = text;
        el.appendChild(p);
    }

    function renderImages(el, images) {
        if (!images || !images.length) return;
        const row = document.createElement("div");
        row.className = "chat-thumbs";
        images.forEach(function (image) {
            const a = document.createElement("a");
            a.href = image.urls.page;
            a.title = image.filename_stem;
            const img = document.createElement("img");
            img.loading = "lazy";
            img.src = image.urls.thumb;
            img.alt = image.filename_stem;
            a.appendChild(img);
            row.appendChild(a);
        });
        el.appendChild(row);
    }

    function renderSteps(el, data) {
        if (!data.steps || !data.steps.length) return;
        const details = document.createElement("details");
        details.className = "chat-steps";
        const summary = document.createElement("summary");
        summary.textContent =
            data.steps.length + " step" + (data.steps.length > 1 ? "s" : "") +
            " · " + data.rounds + " round" + (data.rounds > 1 ? "s" : "") +
            (data.truncated ? " · truncated" : "");
        details.appendChild(summary);
        data.steps.forEach(function (step) {
            const line = document.createElement("div");
            line.textContent = step.summary;
            details.appendChild(line);
        });
        el.appendChild(details);
    }

    function setBusy(busy) {
        input.disabled = busy;
        sendButton.disabled = busy;
        spinner.classList.toggle("d-none", !busy);
        sendIcon.classList.toggle("d-none", busy);
    }

    async function send(question) {
        history.push({role: "user", content: question});
        history = history.slice(-HISTORY_LIMIT);
        renderText(bubble("user"), question);

        const placeholder = bubble("assistant");
        renderText(placeholder, "thinking…");
        setBusy(true);
        try {
            const response = await fetch("/api/ask", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({messages: history}),
            });
            const data = await response.json();
            placeholder.replaceChildren();
            if (!response.ok) {
                placeholder.classList.add("error");
                renderText(placeholder, data.error || ("request failed (" + response.status + ")"));
                return;
            }
            renderText(placeholder, data.answer.text || "(no answer)");
            renderImages(placeholder, data.answer.images);
            renderSteps(placeholder, data);
            history.push({
                role: "assistant",
                content: data.answer.text || "",
                image_stems: data.answer.images.map(function (image) {
                    return image.filename_stem;
                }),
            });
        } catch (error) {
            placeholder.replaceChildren();
            placeholder.classList.add("error");
            renderText(placeholder, "network error: " + error);
        } finally {
            setBusy(false);
            input.focus();
        }
    }

    form.addEventListener("submit", function (event) {
        event.preventDefault();
        const question = input.value.trim();
        if (!question || input.disabled) return;
        input.value = "";
        send(question);
    });

    input.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            form.requestSubmit();
        }
    });

    resetButton.addEventListener("click", function () {
        history = [];
        log.replaceChildren();
        input.focus();
    });

    document.querySelectorAll(".example-prompt").forEach(function (button) {
        button.addEventListener("click", function () {
            input.value = button.textContent;
            input.focus();
        });
    });
})();
