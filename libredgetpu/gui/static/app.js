// LibrEdgeTPU Interactive GUI Frontend Logic

// State
let isDragging = false;
let dragStartPos = null;
const DRAG_THRESHOLD = 5;  // Pixels of movement before treating as drag
let paramsDirty = false;   // Track whether params have been changed

// DOM Elements
const videoStream = document.getElementById('video-stream');
const algorithmSelect = document.getElementById('algorithm-select');
const modeBadge = document.getElementById('mode-badge');
const metricFps = document.getElementById('metric-fps');
const metricLatency = document.getElementById('metric-latency');
const outputArea = document.getElementById('output-area');
const pauseBtn = document.getElementById('pause-btn');
const screenshotBtn = document.getElementById('screenshot-btn');
const paramsSection = document.getElementById('params-section');
const paramsContainer = document.getElementById('params-container');
const applyBtn = document.getElementById('apply-btn');
const paramsStatus = document.getElementById('params-status');

// Get mouse coordinates relative to the image in actual pixel space
function getMousePos(event) {
    const rect = videoStream.getBoundingClientRect();
    const scaleX = videoStream.naturalWidth / rect.width;
    const scaleY = videoStream.naturalHeight / rect.height;

    return {
        x: Math.round((event.clientX - rect.left) * scaleX),
        y: Math.round((event.clientY - rect.top) * scaleY)
    };
}

// Distance between two points
function distance(p1, p2) {
    return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
}

// Mouse event handlers on the video stream image
videoStream.addEventListener('mousedown', (e) => {
    e.preventDefault();
    dragStartPos = getMousePos(e);
    isDragging = false;  // Not dragging yet, just pressed
});

videoStream.addEventListener('mousemove', (e) => {
    if (dragStartPos) {
        const pos = getMousePos(e);
        const dist = distance(dragStartPos, pos);

        if (!isDragging && dist > DRAG_THRESHOLD) {
            // Start dragging
            isDragging = true;
            fetch('/drag_start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(dragStartPos)
            });
        }

        if (isDragging) {
            fetch('/drag_move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(pos)
            });
        }
    }
});

videoStream.addEventListener('mouseup', (e) => {
    e.preventDefault();
    const pos = getMousePos(e);

    if (isDragging) {
        // End drag
        fetch('/drag_end', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(pos)
        });
    } else if (dragStartPos) {
        // Simple click (no significant movement)
        fetch('/click', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(pos)
        });
    }

    isDragging = false;
    dragStartPos = null;
});

videoStream.addEventListener('mouseleave', () => {
    if (isDragging) {
        isDragging = false;
    }
    dragStartPos = null;
});

// Prevent default drag behavior on the image (browser tries to drag-and-drop it)
videoStream.addEventListener('dragstart', (e) => e.preventDefault());

// --- Parameter Controls ---

function renderParamControls(schema, activeParams) {
    if (!schema || schema.length === 0) {
        paramsSection.style.display = 'none';
        return;
    }
    paramsSection.style.display = '';
    paramsContainer.innerHTML = '';
    paramsDirty = false;
    applyBtn.disabled = true;
    paramsStatus.textContent = '';

    schema.forEach(param => {
        const value = (activeParams && activeParams[param.name] !== undefined)
            ? activeParams[param.name]
            : param.default;
        const row = document.createElement('div');
        row.className = 'param-row';

        const label = document.createElement('label');
        label.className = 'param-label';
        label.textContent = param.label;
        label.title = param.description || '';
        row.appendChild(label);

        const controlWrap = document.createElement('div');
        controlWrap.className = 'param-control';

        if (param.type === 'select') {
            const sel = document.createElement('select');
            sel.dataset.paramName = param.name;
            sel.dataset.paramType = 'select';
            param.options.forEach(opt => {
                const o = document.createElement('option');
                o.value = opt;
                o.textContent = opt;
                if (String(opt) === String(value)) o.selected = true;
                sel.appendChild(o);
            });
            sel.addEventListener('change', () => markDirty());
            controlWrap.appendChild(sel);

        } else if (param.type === 'number') {
            const inp = document.createElement('input');
            inp.type = 'number';
            inp.dataset.paramName = param.name;
            inp.dataset.paramType = 'number';
            inp.value = value;
            if (param.min !== undefined) inp.min = param.min;
            if (param.max !== undefined) inp.max = param.max;
            if (param.step !== undefined) inp.step = param.step;
            inp.addEventListener('input', () => markDirty());
            controlWrap.appendChild(inp);

        } else if (param.type === 'range') {
            const inp = document.createElement('input');
            inp.type = 'range';
            inp.dataset.paramName = param.name;
            inp.dataset.paramType = 'range';
            inp.value = value;
            inp.min = param.min !== undefined ? param.min : 0;
            inp.max = param.max !== undefined ? param.max : 1;
            inp.step = param.step !== undefined ? param.step : 0.01;
            const valSpan = document.createElement('span');
            valSpan.className = 'range-value';
            valSpan.textContent = value;
            inp.addEventListener('input', () => {
                valSpan.textContent = inp.value;
                markDirty();
            });
            controlWrap.appendChild(inp);
            controlWrap.appendChild(valSpan);

        } else if (param.type === 'checkbox') {
            const inp = document.createElement('input');
            inp.type = 'checkbox';
            inp.dataset.paramName = param.name;
            inp.dataset.paramType = 'checkbox';
            inp.checked = !!value;
            inp.addEventListener('change', () => markDirty());
            controlWrap.appendChild(inp);
        }

        if (param.description) {
            const desc = document.createElement('div');
            desc.className = 'param-description';
            desc.textContent = param.description;
            controlWrap.appendChild(desc);
        }

        row.appendChild(controlWrap);
        paramsContainer.appendChild(row);
    });
}

function markDirty() {
    paramsDirty = true;
    applyBtn.disabled = false;
    paramsStatus.textContent = '';
    paramsStatus.className = 'params-status';
}

function collectParams() {
    const params = {};
    paramsContainer.querySelectorAll('[data-param-name]').forEach(el => {
        const name = el.dataset.paramName;
        const type = el.dataset.paramType;
        if (type === 'checkbox') {
            params[name] = el.checked;
        } else if (type === 'range') {
            params[name] = parseFloat(el.value);
        } else if (type === 'number') {
            // Send as string so backend can decide int vs float
            params[name] = el.value;
        } else {
            params[name] = el.value;
        }
    });
    return params;
}

applyBtn.addEventListener('click', () => {
    const algorithm = algorithmSelect.value;
    const params = collectParams();

    applyBtn.disabled = true;
    paramsStatus.textContent = 'Applying...';
    paramsStatus.className = 'params-status';

    fetch('/apply_params', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ algorithm, params })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            paramsStatus.textContent = data.error;
            paramsStatus.className = 'params-status params-error';
            applyBtn.disabled = false;
        } else {
            paramsStatus.textContent = 'Applied successfully';
            paramsStatus.className = 'params-status params-success';
            paramsDirty = false;
            // Update displayed values from server response
            if (data.params) {
                paramsContainer.querySelectorAll('[data-param-name]').forEach(el => {
                    const name = el.dataset.paramName;
                    if (data.params[name] !== undefined) {
                        if (el.type === 'checkbox') {
                            el.checked = !!data.params[name];
                        } else {
                            el.value = data.params[name];
                        }
                        // Update range display
                        if (el.dataset.paramType === 'range') {
                            const valSpan = el.parentElement.querySelector('.range-value');
                            if (valSpan) valSpan.textContent = data.params[name];
                        }
                    }
                });
            }
        }
    })
    .catch(err => {
        paramsStatus.textContent = `Network error: ${err.message}`;
        paramsStatus.className = 'params-status params-error';
        applyBtn.disabled = false;
    });
});

// Algorithm selection
algorithmSelect.addEventListener('change', (e) => {
    const algorithm = e.target.value;

    fetch('/switch_algorithm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ algorithm })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('Error switching algorithm:', data.error);
            outputArea.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
        } else {
            console.log('Switched to algorithm:', data.algorithm);
            outputArea.innerHTML = `<p class="hint">Switched to ${data.algorithm}</p>`;
            renderParamControls(data.schema, data.params);
        }
    })
    .catch(err => {
        console.error('Network error:', err);
        outputArea.innerHTML = `<p style="color: red;">Network error: ${err.message}</p>`;
    });
});

// Server-Sent Events for metrics
const metricsSource = new EventSource('/metrics');

metricsSource.onmessage = (event) => {
    try {
        const metrics = JSON.parse(event.data);

        // Update FPS
        if (metrics.fps !== undefined) {
            metricFps.textContent = metrics.fps.toFixed(1);
        }

        // Update latency
        if (metrics.latency_ms !== undefined) {
            metricLatency.textContent = `${metrics.latency_ms.toFixed(2)} ms`;
        }

        // Update mode badge
        if (metrics.mode) {
            modeBadge.textContent = metrics.mode;
            if (metrics.mode === 'SYNTHETIC') {
                modeBadge.classList.add('synthetic');
            } else {
                modeBadge.classList.remove('synthetic');
            }
        }

        // Update output area with algorithm-specific metrics
        updateOutputArea(metrics);

    } catch (err) {
        console.error('Error parsing metrics:', err);
    }
};

metricsSource.onerror = (err) => {
    console.error('SSE error:', err);
    modeBadge.textContent = 'DISCONNECTED';
    modeBadge.style.background = '#ff4444';
};

// Update output area with algorithm-specific metrics
function updateOutputArea(metrics) {
    let html = `<p><strong>Algorithm:</strong> ${metrics.algorithm || 'N/A'}</p>`;
    html += `<p><strong>Mode:</strong> ${metrics.mode || 'N/A'}</p>`;

    // Add algorithm-specific metrics
    for (const [key, value] of Object.entries(metrics)) {
        if (!['fps', 'algorithm', 'mode', 'latency_ms'].includes(key)) {
            const displayKey = key.replace(/_/g, ' ');
            html += `<p><strong>${displayKey}:</strong> ${value}</p>`;
        }
    }

    outputArea.innerHTML = html;
}

// Pause/Play button
let isPaused = false;
pauseBtn.addEventListener('click', () => {
    fetch('/pause', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            isPaused = data.paused;
            pauseBtn.textContent = isPaused ? '▶️ Play' : '⏸️ Pause';
            pauseBtn.classList.toggle('btn-active', isPaused);
        }
    })
    .catch(error => console.error('Pause toggle failed:', error));
});

// Keyboard shortcut: Spacebar to pause/play (skip when focused on form controls)
document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) {
        e.preventDefault();
        pauseBtn.click();
    }
});

// Screenshot button
screenshotBtn.addEventListener('click', () => {
    fetch('/screenshot')
        .then(response => {
            if (!response.ok) throw new Error('Screenshot failed');
            return response.blob();
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const now = new Date();
            const ts = now.toISOString().replace(/[:-]/g, '').replace('T', '_').slice(0, 15);
            a.download = `libredgetpu_screenshot_${ts}.jpg`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        })
        .catch(err => {
            console.error('Screenshot error:', err);
        });
});

// Fetch initial param schema for the default algorithm on page load
fetch(`/param_schema?algorithm=${algorithmSelect.value}`)
    .then(response => response.json())
    .then(data => {
        if (data.schema) {
            renderParamControls(data.schema, data.params);
        }
    })
    .catch(err => console.error('Failed to load initial param schema:', err));

// Initialize
console.log('LibrEdgeTPU Interactive GUI initialized');
