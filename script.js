// 🌌 QUANTUM NEXUS - Advanced BB84 Simulator JavaScript
// Enhanced with particle effects, quantum animations, and modern UI interactions

// State variables for the simulation
let simulationData = [];
let simulationInterval = null;
let isContinuous = false;
let particles = [];

// DOM element references
const aliceRandomRadio = document.getElementById('alice-random');
const aliceFixedRadio = document.getElementById('alice-fixed');
const aliceFixedSelect = document.getElementById('alice-fixed-basis');
const bobRandomRadio = document.getElementById('bob-random');
const bobFixedRadio = document.getElementById('bob-fixed');
const bobFixedSelect = document.getElementById('bob-fixed-basis');
const eveToggle = document.getElementById('eve-toggle');

const singlePhotonBtn = document.getElementById('single-photon-btn');
const continuousBtn = document.getElementById('continuous-btn');
const stopBtn = document.getElementById('stop-btn');
const fastForwardBtn = document.getElementById('fast-forward-btn');
const resetBtn = document.getElementById('reset-btn');

const aliceBasisViz = document.getElementById('alice-basis-viz');
const aliceBitDisplay = document.getElementById('alice-bit');
const bobBasisViz = document.getElementById('bob-basis-viz');
const bobBitDisplay = document.getElementById('bob-bit');
const photon = document.getElementById('photon');
const evesZone = document.getElementById('eves-zone');
const photonPath = document.querySelector('.transmission-channel');
const simulationStatusDisplay = document.getElementById('simulation-status');

const resultsTableBody = document.getElementById('results-table-body');
const siftedKeyAliceDisplay = document.getElementById('sifted-key-alice');
const siftedKeyBobDisplay = document.getElementById('sifted-key-bob');
const totalPhotonsDisplay = document.getElementById('total-photons');
const keyBitsDisplay = document.getElementById('key-bits');
const errorCountDisplay = document.getElementById('error-count');
const errorRateDisplay = document.getElementById('error-rate');
const simulationVizSection = document.querySelector('.quantum-visualization-chamber');

// Constants for bases and their visual representations
const H_V = "H/V";
const PLUS_MINUS_45 = "+45/-45";
const BASES = [H_V, PLUS_MINUS_45];
const BASE_SYMBOLS = {
    "H/V": ["H", "V"], // Represents bits 0 and 1 in the rectilinear basis
    "+45/-45": ["+45", "-45"], // Represents bits 0 and 1 in the diagonal basis
};

/**
 * Initialize the quantum interface with particle effects
 */
const initializeQuantumInterface = () => {
    createParticles();
    hideLoadingOverlay();
    addQuantumEffects();
    updateEveZoneAppearance();
};

/**
 * Create animated background particles
 */
const createParticles = () => {
    const particlesContainer = document.getElementById('particles-bg');
    const particleCount = 50;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 6 + 's';
        particle.style.animationDuration = (Math.random() * 3 + 3) + 's';
        particlesContainer.appendChild(particle);
    }
};

/**
 * Hide the quantum loading overlay
 */
const hideLoadingOverlay = () => {
    const loadingOverlay = document.getElementById('quantum-loading');
    if (loadingOverlay) {
        loadingOverlay.style.opacity = '0';
        setTimeout(() => {
            loadingOverlay.style.display = 'none';
        }, 500);
    }
};

/**
 * Add quantum effects to the interface
 */
const addQuantumEffects = () => {
    // Add hover effects to quantum stations
    const stations = document.querySelectorAll('.quantum-station');
    stations.forEach(station => {
        station.addEventListener('mouseenter', () => {
            station.style.transform = 'translateY(-5px) scale(1.02)';
        });
        
        station.addEventListener('mouseleave', () => {
            station.style.transform = 'translateY(0) scale(1)';
        });
    });
    
    // Add ripple effects to buttons
    const buttons = document.querySelectorAll('.quantum-btn');
    buttons.forEach(button => {
        button.addEventListener('click', createRippleEffect);
    });
};

/**
 * Create ripple effect on button click
 */
const createRippleEffect = (event) => {
    const button = event.currentTarget;
    const ripple = document.createElement('span');
    const rect = button.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = event.clientX - rect.left - size / 2;
    const y = event.clientY - rect.top - size / 2;
    
    ripple.style.width = ripple.style.height = size + 'px';
    ripple.style.left = x + 'px';
    ripple.style.top = y + 'px';
    ripple.classList.add('ripple');
    
    button.appendChild(ripple);
    
    setTimeout(() => {
        ripple.remove();
    }, 600);
};

/**
 * Generates a random bit (0 or 1).
 * @returns {number} The generated bit.
 */
const getRandomBit = () => Math.round(Math.random());

/**
 * Generates a random basis.
 * @returns {string} The randomly selected basis.
 */
const getRandomBasis = () => {
    return Math.random() < 0.5 ? H_V : PLUS_MINUS_45;
};

/**
 * Gets the bit value from a basis and a symbol.
 * @param {string} basis - The basis used.
 * @param {string} symbol - The symbol representing the polarization state.
 * @returns {number} The corresponding bit (0 or 1).
 */
const getBitFromSymbol = (basis, symbol) => {
    const symbols = BASE_SYMBOLS[basis];
    return symbols.indexOf(symbol);
};

/**
 * Simulates a measurement of a photon given the measurement basis.
 * The core logic of BB84: if bases don't match, the outcome is random.
 * @param {object} photon - The photon object {bit, basis}.
 * @param {string} measurementBasis - The basis used for measurement.
 * @returns {{bit: number, outcomeBasis: string}} The measured bit and its corresponding symbol.
 */
const measurePhoton = (photon, measurementBasis) => {
    if (photon.basis === measurementBasis) {
        // Bases match, outcome is certain and correct
        return { bit: photon.bit, outcomeBasis: BASE_SYMBOLS[measurementBasis][photon.bit] };
    } else {
        // Bases don't match, outcome is random (50/50 chance)
        const randomBit = Math.round(Math.random());
        return { bit: randomBit, outcomeBasis: BASE_SYMBOLS[measurementBasis][randomBit] };
    }
};

/**
 * Enhanced photon transmission with quantum effects
 */
const sendPhoton = () => {
    const aliceBasisMode = document.querySelector('input[name="alice-basis-mode"]:checked').value;
    const bobBasisMode = document.querySelector('input[name="bob-basis-mode"]:checked').value;
    const eveEnabled = eveToggle.checked;

    // Alice's actions: Choose basis and bit, prepare photon
    const aliceBasis = aliceBasisMode === 'random' ? getRandomBasis() : aliceFixedSelect.value;
    const aliceBit = getRandomBit();
    
    // Display Alice's initial state with quantum effects
    updateAliceDisplay(aliceBasis, aliceBit);
    resetBobDisplay();
    
    // Reset photon for new animation
    resetPhoton();
    const initialPhotonSymbol = BASE_SYMBOLS[aliceBasis][aliceBit];
    updatePhotonDisplay(initialPhotonSymbol);
    
    // Update status and activate quantum effects
    updateSimulationStatus("Photon in quantum transit...", "transmitting");
    activateQuantumPath();

    // Start the enhanced visual animation
    requestAnimationFrame(() => {
        animatePhotonTransmission();
    });

    // Handle Eve's interception (if enabled)
    setTimeout(() => {
        let finalPhotonBit = aliceBit;
        let finalPhotonBasis = aliceBasis;
        let eveOutcome = '--';
        let eveBasis = '--';
        
        if (eveEnabled) {
            // Eve's actions: Choose random basis, measure the photon
            eveBasis = getRandomBasis();
            const { bit: eveBit, outcomeBasis: eveOutcomeTemp } = measurePhoton({ bit: aliceBit, basis: aliceBasis }, eveBasis);
            
            // The photon's state is collapsed and re-prepared based on Eve's measurement
            finalPhotonBit = eveBit;
            finalPhotonBasis = eveBasis;
            eveOutcome = eveOutcomeTemp;
            
            // Visually update the photon for Bob's reception
            const newPhotonSymbol = BASE_SYMBOLS[finalPhotonBasis][finalPhotonBit];
            updatePhotonDisplay(newPhotonSymbol);
            
            // Show Eve's warning
            showEveWarning();
        }

        // Bob's actions: Choose basis and measure the arriving photon
        const bobBasis = bobBasisMode === 'random' ? getRandomBasis() : bobFixedSelect.value;
        const { bit: bobBit, outcomeBasis: bobOutcome } = measurePhoton({ bit: finalPhotonBit, basis: finalPhotonBasis }, bobBasis);
        
        // Display Bob's results after the animation
        setTimeout(() => {
            updateBobDisplay(bobBasis, bobBit);
            
            // Start fade out animation and add to the data table
            setTimeout(() => {
                fadeOutPhoton();
                
                // Collect and store the full simulation data for this run
                const simulationRow = {
                    photon: simulationData.length + 1,
                    aliceBasis: aliceBasis,
                    aliceValue: aliceBit,
                    eveBasis: eveBasis,
                    eveOutcome: eveOutcome,
                    bobBasis: bobBasis,
                    bobOutcome: bobOutcome,
                    basesMatch: aliceBasis === bobBasis,
                    keyBit: (aliceBasis === bobBasis) ? bobBit : '--'
                };
                
                simulationData.push(simulationRow);
                
                // Update UI tables and statistics
                updateTableAndStats();
                
                // Re-enable buttons if not in continuous mode
                enableButtons(); 
                
                // Reset status and path
                if (!isContinuous) {
                    updateSimulationStatus("Quantum Matrix Ready", "ready");
                    deactivateQuantumPath();
                }
            }, 500); // Wait for the fade out to begin
        }, 1000);
    }, 1000); // Time for the photon to travel to Eve/Bob
};

/**
 * Update Alice's display with quantum effects
 */
const updateAliceDisplay = (basis, bit) => {
    aliceBasisViz.textContent = basis === H_V ? 'H/V' : '±45';
    aliceBitDisplay.textContent = bit;
    
    // Add quantum animation
    aliceBasisViz.style.animation = 'none';
    aliceBasisViz.offsetHeight; // Trigger reflow
    aliceBasisViz.style.animation = 'quantum-pulse 0.5s ease-in-out';
};

/**
 * Reset Bob's display
 */
const resetBobDisplay = () => {
    bobBasisViz.textContent = '?';
    bobBitDisplay.textContent = '?';
};

/**
 * Update Bob's display with quantum effects
 */
const updateBobDisplay = (basis, bit) => {
    bobBasisViz.textContent = basis === H_V ? 'H/V' : '±45';
    bobBitDisplay.textContent = bit;
    
    // Add quantum animation
    bobBasisViz.style.animation = 'none';
    bobBasisViz.offsetHeight; // Trigger reflow
    bobBasisViz.style.animation = 'quantum-pulse 0.5s ease-in-out';
};

/**
 * Reset photon for new animation
 */
const resetPhoton = () => {
    photon.classList.remove('arrived');
    photon.classList.remove('fade-out');
    photon.style.left = '0%';
    photon.style.opacity = '1';
};

/**
 * Update photon display
 */
const updatePhotonDisplay = (symbol) => {
    const photonLabel = photon.querySelector('.photon-label');
    if (photonLabel) {
        photonLabel.textContent = symbol;
    }
};

/**
 * Update simulation status with enhanced styling
 */
const updateSimulationStatus = (text, state) => {
    const statusText = simulationStatusDisplay.querySelector('.status-text');
    const statusIcon = simulationStatusDisplay.querySelector('.status-icon');
    
    if (statusText) statusText.textContent = text;
    
    // Update status styling based on state
    simulationStatusDisplay.className = `simulation-status ${state}`;
    
    // Update icon based on state
    if (statusIcon) {
        switch (state) {
            case 'transmitting':
                statusIcon.textContent = '⚛';
                break;
            case 'ready':
                statusIcon.textContent = '⚡';
                break;
            default:
                statusIcon.textContent = '⚡';
        }
    }
};

/**
 * Activate quantum path with enhanced effects
 */
const activateQuantumPath = () => {
    const channelPath = document.querySelector('.channel-path');
    if (channelPath) {
        channelPath.style.boxShadow = '0 0 30px var(--quantum-primary)';
        channelPath.style.animation = 'quantum-path-active 2s ease-in-out infinite';
    }
};

/**
 * Deactivate quantum path
 */
const deactivateQuantumPath = () => {
    const channelPath = document.querySelector('.channel-path');
    if (channelPath) {
        channelPath.style.boxShadow = '0 0 20px var(--quantum-primary)';
        channelPath.style.animation = 'none';
    }
};

/**
 * Animate photon transmission with quantum effects
 */
const animatePhotonTransmission = () => {
    photon.classList.add('arrived');
    
    // Add quantum trail effect
    createPhotonTrail();
};

/**
 * Create photon trail effect
 */
const createPhotonTrail = () => {
    const trail = document.createElement('div');
    trail.className = 'photon-trail';
    trail.style.position = 'absolute';
    trail.style.width = '4px';
    trail.style.height = '4px';
    trail.style.background = 'var(--quantum-primary)';
    trail.style.borderRadius = '50%';
    trail.style.left = '0%';
    trail.style.top = '50%';
    trail.style.transform = 'translateY(-50%)';
    trail.style.animation = 'trail-fade 1s ease-out forwards';
    
    photonPath.appendChild(trail);
    
    setTimeout(() => {
        trail.remove();
    }, 1000);
};

/**
 * Fade out photon with quantum effects
 */
const fadeOutPhoton = () => {
    photon.classList.add('fade-out');
    
    // Add quantum collapse effect
    createQuantumCollapse();
};

/**
 * Create quantum collapse effect
 */
const createQuantumCollapse = () => {
    const collapse = document.createElement('div');
    collapse.className = 'quantum-collapse';
    collapse.style.position = 'absolute';
    collapse.style.width = '100px';
    collapse.style.height = '100px';
    collapse.style.border = '2px solid var(--quantum-primary)';
    collapse.style.borderRadius = '50%';
    collapse.style.left = '50%';
    collapse.style.top = '50%';
    collapse.style.transform = 'translate(-50%, -50%)';
    collapse.style.animation = 'collapse-expand 0.5s ease-out forwards';
    
    photonPath.appendChild(collapse);
    
    setTimeout(() => {
        collapse.remove();
    }, 500);
};

/**
 * Show Eve's warning with enhanced styling
 */
const showEveWarning = () => {
    const warning = document.getElementById('eve-warning');
    if (warning) {
        warning.classList.add('show');
        warning.style.animation = 'warning-pulse 2s ease-in-out infinite';
    }
};

/**
 * Starts a continuous loop of photon transmissions.
 */
const runContinuous = () => {
    isContinuous = true;
    disableButtons();
    stopBtn.style.display = 'inline-block';
    
    updateSimulationStatus("Continuous quantum stream active...", "transmitting");
    activateQuantumPath();
    
    // Set a recurring interval to send photons
    simulationInterval = setInterval(() => {
        sendPhoton();
    }, 3500); // Increased interval to allow for full animation and reset
};

/**
 * Stops the continuous photon transmission.
 */
const stopContinuous = () => {
    isContinuous = false;
    clearInterval(simulationInterval);
    enableButtons();
    stopBtn.style.display = 'none';
    
    updateSimulationStatus("Quantum Matrix Ready", "ready");
    deactivateQuantumPath();
};

/**
 * Simulates 100 photon transmissions without animation for fast data generation.
 */
const fastForward = () => {
    // Hide the visualization section as it's not relevant for fast forward
    simulationVizSection.style.display = 'none';
    
    disableButtons();
    updateSimulationStatus("Processing 100 quantum photons...", "processing");

    const numPhotons = 100;
    const eveEnabled = eveToggle.checked;

    for (let i = 0; i < numPhotons; i++) {
        // Simulate each step without animation
        const aliceBasisMode = document.querySelector('input[name="alice-basis-mode"]:checked').value;
        const bobBasisMode = document.querySelector('input[name="bob-basis-mode"]:checked').value;
        
        const aliceBasis = aliceBasisMode === 'random' ? getRandomBasis() : aliceFixedSelect.value;
        const aliceBit = getRandomBit();
        let finalPhotonBit = aliceBit;
        let finalPhotonBasis = aliceBasis;
        let eveOutcome = '--';
        let eveBasis = '--';
        
        if (eveEnabled) {
            eveBasis = getRandomBasis();
            const { bit: eveBit, outcomeBasis: eveOutcomeTemp } = measurePhoton({ bit: aliceBit, basis: aliceBasis }, eveBasis);
            finalPhotonBit = eveBit;
            finalPhotonBasis = eveBasis;
            eveOutcome = eveOutcomeTemp;
        }
        
        const bobBasis = bobBasisMode === 'random' ? getRandomBasis() : bobFixedSelect.value;
        const { bit: bobBit, outcomeBasis: bobOutcome } = measurePhoton({ bit: finalPhotonBit, basis: finalPhotonBasis }, bobBasis);

        // Push data to the simulation log
        simulationData.push({
            photon: simulationData.length + 1,
            aliceBasis: aliceBasis,
            aliceValue: aliceBit,
            eveBasis: eveBasis,
            eveOutcome: eveOutcome,
            bobBasis: bobBasis,
            bobOutcome: bobOutcome,
            basesMatch: aliceBasis === bobBasis,
            keyBit: (aliceBasis === bobBasis) ? bobBit : '--'
        });
    }
    
    // Update UI after all simulations are complete
    updateTableAndStats();
    enableButtons();
    updateSimulationStatus("Quantum burst complete. Results for 100 photons displayed.", "ready");
};

/**
 * Resets all simulation data and the UI to its initial state.
 */
const resetSimulation = () => {
    stopContinuous();
    simulationData = [];
    resultsTableBody.innerHTML = '';
    siftedKeyAliceDisplay.textContent = '--';
    siftedKeyBobDisplay.textContent = '--';
    totalPhotonsDisplay.textContent = '0';
    keyBitsDisplay.textContent = '0';
    errorCountDisplay.textContent = '0';
    errorRateDisplay.textContent = '0.00%';
    aliceBitDisplay.textContent = '?';
    aliceBasisViz.textContent = '?';
    bobBitDisplay.textContent = '?';
    bobBasisViz.textContent = '?';
    
    resetPhoton();
    evesZone.classList.remove('active');
    simulationVizSection.style.display = 'block'; // Show visualization again
    updateSimulationStatus("Quantum Matrix Ready", "ready");
    deactivateQuantumPath();
    
    // Hide Eve's warning
    const warning = document.getElementById('eve-warning');
    if (warning) {
        warning.classList.remove('show');
    }
    
    enableButtons();
};

/**
 * Updates the data table and the key/error statistics based on simulationData.
 */
const updateTableAndStats = () => {
    // Clear the existing table rows
    resultsTableBody.innerHTML = '';

    let keyAlice = [];
    let keyBob = [];
    let errors = 0;
    let keyBits = 0;

    // Loop through the simulation data to populate the table and calculate stats
    simulationData.forEach(row => {
        const newRow = resultsTableBody.insertRow();
        newRow.className = row.basesMatch ? 'key-match' : 'no-key-match';
        newRow.innerHTML = `
            <td>${row.photon}</td>
            <td>${row.aliceBasis}</td>
            <td>${row.aliceValue}</td>
            <td>${row.eveBasis}</td>
            <td>${row.eveOutcome}</td>
            <td>${row.bobBasis}</td>
            <td>${row.bobOutcome}</td>
            <td>${row.basesMatch ? 'YES' : 'NO'}</td>
            <td>${row.keyBit}</td>
        `;

        // Sifting the key
        if (row.basesMatch) {
            keyBits++;
            keyAlice.push(row.aliceValue);
            keyBob.push(row.keyBit);
            if (row.aliceValue !== row.keyBit) {
                errors++;
            }
        }
    });
    
    // Update the final key and error displays
    siftedKeyAliceDisplay.textContent = keyAlice.join('');
    siftedKeyBobDisplay.textContent = keyBob.join('');
    totalPhotonsDisplay.textContent = simulationData.length;
    keyBitsDisplay.textContent = keyBits;
    errorCountDisplay.textContent = errors;
    const errorRate = keyBits > 0 ? (errors / keyBits) * 100 : 0;
    errorRateDisplay.textContent = `${errorRate.toFixed(2)}%`;
    
    // Update error status indicator
    updateErrorStatus(errors, keyBits);
};

/**
 * Update error status indicator
 */
const updateErrorStatus = (errors, keyBits) => {
    const errorStatus = document.getElementById('error-status');
    if (errorStatus) {
        if (errors > 0) {
            errorStatus.style.background = 'var(--quantum-danger)';
            errorStatus.style.animation = 'status-pulse 1s ease-in-out infinite';
        } else {
            errorStatus.style.background = 'var(--quantum-success)';
            errorStatus.style.animation = 'status-pulse 2s ease-in-out infinite';
        }
    }
};

// ===== Backend integration (FastAPI + WebSocket) =====
const BACKEND_BASE = (location.hostname === 'localhost' || location.hostname === '127.0.0.1')
  ? 'http://127.0.0.1:8000'
  : (window.QKD_BACKEND || 'http://127.0.0.1:8000');

let ws;

function connectStream() {
  try {
    const wsBase = BACKEND_BASE.startsWith('https')
      ? BACKEND_BASE.replace('https', 'wss')
      : BACKEND_BASE.replace('http', 'ws');
    ws = new WebSocket(wsBase + '/qkd/stream');
    ws.onopen = () => updateSimulationStatus('Connected to QKD backend', 'ready');
    ws.onclose = () => updateSimulationStatus('Stream closed', 'ready');
    ws.onerror = () => updateSimulationStatus('Stream error', 'ready');
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'status') {
        updateSimulationStatus(msg.message, 'processing');
      } else if (msg.type === 'job_submitted') {
        // no-op, but could show progress
      } else if (msg.type === 'trial_result') {
        // Append row to table
        const row = {
          photon: msg.index + 1,
          aliceBasis: msg.alice_basis === 1 ? '+45/-45' : 'H/V',
          aliceValue: msg.alice_bit,
          eveBasis: (msg.eve_basis === null || msg.eve_basis === undefined) ? '--' : (msg.eve_basis === 1 ? '+45/-45' : 'H/V'),
          eveOutcome: '--',
          bobBasis: msg.bob_basis === 1 ? '+45/-45' : 'H/V',
          bobOutcome: msg.bob_measured,
          basesMatch: msg.alice_basis === msg.bob_basis,
          keyBit: (msg.alice_basis === msg.bob_basis) ? msg.bob_measured : '--'
        };
        simulationData.push(row);
        updateTableAndStats();
      } else if (msg.type === 'summary') {
        siftedKeyAliceDisplay.textContent = msg.alice_sifted || '--';
        siftedKeyBobDisplay.textContent = msg.bob_sifted || '--';
        errorRateDisplay.textContent = (typeof msg.qber === 'number') ? `${msg.qber.toFixed(2)}%` : '--';
        keyBitsDisplay.textContent = msg.length;
        updateSimulationStatus('QKD run complete', 'ready');
      } else if (msg.type === 'error') {
        updateSimulationStatus(`Error: ${msg.message}`, 'ready');
      } else if (msg.type === 'done') {
        // done
      }
    };
  } catch (e) {
    // ignore
  }
}

async function startQKDBackend({ n_trials = 20, with_eve = false, shots_per_job = 1, instance = null } = {}) {
  try {
    const body = { n_trials, with_eve, shots_per_job, instance };
    if (window.__IBM_TOKEN_FOR_RUN__) body.token = window.__IBM_TOKEN_FOR_RUN__;
    const res = await fetch(BACKEND_BASE + '/qkd/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (!res.ok) {
      const txt = await res.text();
      updateSimulationStatus(`Failed to start (${res.status}): ${txt}`, 'ready');
    }
  } catch (e) {
    updateSimulationStatus('Failed to start backend run', 'ready');
  }
}

// Bind UI: use backend when Fast Forward is clicked holding Shift key
fastForwardBtn.addEventListener('click', (e) => {
  if (e.shiftKey) {
    // Backend-driven QKD (Shift+Click)
    resetSimulation();
    connectStream();
    const withEve = document.getElementById('eve-toggle').checked;
    startQKDBackend({ n_trials: 50, with_eve: withEve, shots_per_job: 1 });
  } else {
    // Local fast simulation
    fastForward();
  }
});

// ===== IBM Modal bindings =====
const ibmModal = document.getElementById('ibm-modal');
const ibmBtn = document.getElementById('run-ibm-btn');
const ibmStart = document.getElementById('ibm-start');
const ibmCancel = document.getElementById('ibm-cancel');
const ibmToken = document.getElementById('ibm-token');
const ibmInstance = document.getElementById('ibm-instance');
const ibmTrials = document.getElementById('ibm-trials');
const ibmShots = document.getElementById('ibm-shots');

if (ibmBtn) {
  ibmBtn.addEventListener('click', () => {
    ibmModal.classList.remove('hidden');
    ibmModal.classList.add('flex');
  });
}

if (ibmCancel) {
  ibmCancel.addEventListener('click', () => {
    ibmModal.classList.add('hidden');
    ibmModal.classList.remove('flex');
  });
}

if (ibmStart) {
  ibmStart.addEventListener('click', async () => {
    const token = (ibmToken.value || '').trim();
    const instance = (ibmInstance.value || '').trim();
    const n_trials = Math.max(1, parseInt(ibmTrials.value || '50', 10));
    const shots_per_job = Math.max(1, parseInt(ibmShots.value || '1', 10));
    const with_eve = document.getElementById('eve-toggle').checked;

    window.__IBM_TOKEN_FOR_RUN__ = token || undefined;

    resetSimulation();
    connectStream();
    await startQKDBackend({ n_trials, with_eve, shots_per_job, instance: instance || null });

    ibmModal.classList.add('hidden');
    ibmModal.classList.remove('flex');
  });
}

/**
 * Enables all control buttons.
 */
const enableButtons = () => {
    singlePhotonBtn.disabled = false;
    continuousBtn.disabled = false;
    fastForwardBtn.disabled = false;
    resetBtn.disabled = false;
};

/**
 * Disables all control buttons.
 */
const disableButtons = () => {
    singlePhotonBtn.disabled = true;
    continuousBtn.disabled = true;
    fastForwardBtn.disabled = true;
    resetBtn.disabled = true;
};

/**
 * Updates the visual appearance of Eve's zone based on the toggle state.
 */
const updateEveZoneAppearance = () => {
    const zoneCore = evesZone.querySelector('.zone-core');
    if (eveToggle.checked) {
        evesZone.classList.remove('eavesdropper-zone-disabled');
        evesZone.classList.add('eavesdropper-zone-active');
        if (zoneCore) zoneCore.classList.add('active');
    } else {
        evesZone.classList.remove('eavesdropper-zone-active');
        evesZone.classList.add('eavesdropper-zone-disabled');
        if (zoneCore) zoneCore.classList.remove('active');
    }
};

// --- Event Listeners ---
singlePhotonBtn.addEventListener('click', () => {
    disableButtons();
    sendPhoton();
});

continuousBtn.addEventListener('click', runContinuous);
stopBtn.addEventListener('click', stopContinuous);
fastForwardBtn.addEventListener('click', fastForward);
resetBtn.addEventListener('click', resetSimulation);
eveToggle.addEventListener('change', updateEveZoneAppearance);

// Handle fixed basis selection logic
aliceFixedRadio.addEventListener('change', () => {
    aliceFixedSelect.disabled = !aliceFixedRadio.checked;
});

bobFixedRadio.addEventListener('change', () => {
    bobFixedSelect.disabled = !bobFixedRadio.checked;
});

// Initialize the quantum interface when the page loads
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(initializeQuantumInterface, 1000); // Show loading for 1 second
});

// Add CSS animations dynamically
const addDynamicCSS = () => {
    const style = document.createElement('style');
    style.textContent = `
        @keyframes quantum-pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        
        @keyframes quantum-path-active {
            0%, 100% { box-shadow: 0 0 30px var(--quantum-primary); }
            50% { box-shadow: 0 0 50px var(--quantum-primary); }
        }
        
        @keyframes trail-fade {
            0% { opacity: 1; transform: translateY(-50%) scale(1); }
            100% { opacity: 0; transform: translateY(-50%) scale(0); }
        }
        
        @keyframes collapse-expand {
            0% { transform: translate(-50%, -50%) scale(0); opacity: 1; }
            100% { transform: translate(-50%, -50%) scale(2); opacity: 0; }
        }
        
        .photon-trail {
            z-index: 5;
        }
        
        .quantum-collapse {
            z-index: 15;
        }
        
        .ripple {
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: scale(0);
            animation: ripple-animation 0.6s linear;
            pointer-events: none;
        }
        
        @keyframes ripple-animation {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
};

// Add dynamic CSS when the page loads
document.addEventListener('DOMContentLoaded', addDynamicCSS);
