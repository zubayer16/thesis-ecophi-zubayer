// Global variables
let currentResults = null;
let isLoading = false;

// DOM Elements
const elements = {
    // Controls
    phaseType: document.getElementById('phaseType'),
    phaseNum: document.getElementById('phaseNum'),
    numSamples: document.getElementById('numSamples'),
    plotType: document.getElementById('plotType'),
    generateBtn: document.getElementById('generateBtn'),
    downloadBtn: document.getElementById('downloadBtn'),
    healthCheck: document.getElementById('healthCheck'),
    
    // States
    loadingState: document.getElementById('loadingState'),
    errorState: document.getElementById('errorState'),
    emptyState: document.getElementById('emptyState'),
    resultsArea: document.getElementById('resultsArea'),
    
    // Results
    plotImage: document.getElementById('plotImage'),
    resultsTitle: document.getElementById('resultsTitle'),
    
    // Metrics
    metricsPanel: document.getElementById('metricsPanel'),
    quickStats: document.getElementById('quickStats'),
    
    // Quick stats
    inferenceTime: document.getElementById('inferenceTime'),
    throughput: document.getElementById('throughput'),
    device: document.getElementById('device'),
    
    // Detailed metrics
    totalInferenceTime: document.getElementById('totalInferenceTime'),
    perSampleTime: document.getElementById('perSampleTime'),
    samplesPerSecond: document.getElementById('samplesPerSecond'),
    hardwareDetails: document.getElementById('hardwareDetails'),
    modelPhaseType: document.getElementById('modelPhaseType'),
    modelPhaseNum: document.getElementById('modelPhaseNum'),
    modelSamples: document.getElementById('modelSamples')
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
});

function initializeApp() {
    // Check health on startup
    checkHealth();
    
    // Show empty state initially
    showState('empty');
    
    // Add smooth animations
    document.body.style.opacity = '0';
    setTimeout(() => {
        document.body.style.transition = 'opacity 0.5s ease-in';
        document.body.style.opacity = '1';
    }, 100);
}

function setupEventListeners() {
    // Generate button
    elements.generateBtn.addEventListener('click', generateVisualization);
    
    // Download button
    elements.downloadBtn.addEventListener('click', downloadResults);
    
    // Health check button
    elements.healthCheck.addEventListener('click', checkHealth);
    
    // Form validation
    elements.numSamples.addEventListener('input', validateNumSamples);
    
    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
    
    // Auto-save form state
    setupFormStateManagement();
}

function setupFormStateManagement() {
    // Load saved form state
    loadFormState();
    
    // Save form state on changes
    [elements.phaseType, elements.phaseNum, elements.numSamples, elements.plotType].forEach(element => {
        element.addEventListener('change', saveFormState);
    });
}

function validateNumSamples() {
    const value = parseInt(elements.numSamples.value);
    const min = parseInt(elements.numSamples.min);
    const max = parseInt(elements.numSamples.max);
    
    if (value < min) {
        elements.numSamples.value = min;
        showNotification('Minimum samples is ' + min, 'warning');
    } else if (value > max) {
        elements.numSamples.value = max;
        showNotification('Maximum samples is ' + max, 'warning');
    }
}

function handleKeyboardShortcuts(e) {
    // Ctrl/Cmd + Enter to generate
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (!isLoading) {
            generateVisualization();
        }
    }
    
    // Ctrl/Cmd + D to download
    if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
        e.preventDefault();
        if (currentResults) {
            downloadResults();
        }
    }
    
    // Escape to close modals
    if (e.key === 'Escape') {
        closeAllModals();
    }
}

async function generateVisualization() {
    if (isLoading) return;
    
    try {
        isLoading = true;
        updateGenerateButton(true);
        showState('loading');
        
        const params = new URLSearchParams({
            phase_type: elements.phaseType.value,
            phase_num: elements.phaseNum.value,
            num_samples: elements.numSamples.value,
            plot_type: elements.plotType.value
        });
        
        // Add progress simulation
        simulateProgress();
        
        const response = await fetch(`/visualize?${params}`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        currentResults = result;
        
        // Display results
        displayResults(result);
        showState('results');
        
        // Update quick stats
        updateQuickStats(result);
        
        // Update detailed metrics
        updateDetailedMetrics(result);
        
        // Enable download button
        elements.downloadBtn.disabled = false;
        
        // Show success notification
        showNotification('Visualization generated successfully!', 'success');
        
        // Save form state
        saveFormState();
        
    } catch (error) {
        console.error('Error generating visualization:', error);
        showError(error.message || 'Failed to generate visualization');
        showNotification('Failed to generate visualization', 'error');
    } finally {
        isLoading = false;
        updateGenerateButton(false);
    }
}

function simulateProgress() {
    const progressBar = document.querySelector('.progress-bar');
    if (progressBar) {
        progressBar.style.animation = 'progress 2s ease-in-out infinite';
    }
}

function displayResults(result) {
    elements.plotImage.src = `data:image/png;base64,${result.plot}`;
    elements.plotImage.alt = `${result.phase_type} Phase ${result.phase_num} Visualization`;
    
    elements.resultsTitle.textContent = 
        `${result.phase_type.charAt(0).toUpperCase() + result.phase_type.slice(1)} Phase ${result.phase_num} Results`;
}

function updateQuickStats(result) {
    elements.inferenceTime.textContent = `${(result.avg_inference_time_per_sample * 1000).toFixed(2)} ms`;
    elements.throughput.textContent = `${result.samples_per_second.toFixed(1)} samples/s`;
    elements.device.textContent = result.hardware_info.device || 'Unknown';
    
    elements.quickStats.style.display = 'block';
}

function updateDetailedMetrics(result) {
    // Performance metrics
    elements.totalInferenceTime.textContent = `${(result.total_inference_time * 1000).toFixed(2)} ms`;
    elements.perSampleTime.textContent = `${(result.avg_inference_time_per_sample * 1000).toFixed(2)} ms`;
    elements.samplesPerSecond.textContent = `${result.samples_per_second.toFixed(1)} samples/s`;
    
    // Model info
    elements.modelPhaseType.textContent = result.phase_type.charAt(0).toUpperCase() + result.phase_type.slice(1);
    elements.modelPhaseNum.textContent = result.phase_num;
    elements.modelSamples.textContent = result.num_samples;
    
    // Hardware details
    updateHardwareDetails(result.hardware_info);
    
    // Show metrics panel
    elements.metricsPanel.style.display = 'block';
    
    // Smooth scroll to metrics
    setTimeout(() => {
        elements.metricsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 500);
}

function updateHardwareDetails(hardwareInfo) {
    const container = elements.hardwareDetails;
    container.innerHTML = '';
    
    const importantFields = [
        'cpu_model', 'cpu_cores', 'total_memory', 'gpu_name', 'gpu_memory_total'
    ];
    
    importantFields.forEach(field => {
        if (hardwareInfo[field]) {
            const row = document.createElement('div');
            row.className = 'metric-row';
            
            const label = document.createElement('span');
            label.className = 'metric-label';
            label.textContent = formatFieldName(field);
            
            const value = document.createElement('span');
            value.className = 'metric-value';
            value.textContent = hardwareInfo[field];
            
            row.appendChild(label);
            row.appendChild(value);
            container.appendChild(row);
        }
    });
}

function formatFieldName(fieldName) {
    return fieldName
        .replace(/_/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase());
}

function updateGenerateButton(loading) {
    if (loading) {
        elements.generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
        elements.generateBtn.disabled = true;
    } else {
        elements.generateBtn.innerHTML = '<i class="fas fa-play"></i> Generate Visualization';
        elements.generateBtn.disabled = false;
    }
}

function showState(state) {
    // Hide all states
    elements.loadingState.style.display = 'none';
    elements.errorState.style.display = 'none';
    elements.emptyState.style.display = 'none';
    elements.resultsArea.style.display = 'none';
    
    // Show selected state
    switch (state) {
        case 'loading':
            elements.loadingState.style.display = 'flex';
            break;
        case 'error':
            elements.errorState.style.display = 'flex';
            break;
        case 'empty':
            elements.emptyState.style.display = 'flex';
            break;
        case 'results':
            elements.resultsArea.style.display = 'flex';
            break;
    }
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    showState('error');
}

async function downloadResults() {
    if (!currentResults) {
        showNotification('No results to download', 'warning');
        return;
    }
    
    try {
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
        const filename = `ts_segmentation_results_${timestamp}.json`;
        
        const dataStr = JSON.stringify(currentResults, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        showNotification('Results downloaded successfully!', 'success');
        
    } catch (error) {
        console.error('Error downloading results:', error);
        showNotification('Failed to download results', 'error');
    }
}

async function checkHealth() {
    try {
        const response = await fetch('/health');
        const health = await response.json();
        
        if (health.status === 'healthy') {
            showNotification('System is healthy and ready', 'success');
        } else {
            showNotification('System health check failed', 'warning');
        }
        
        console.log('Health check result:', health);
        
    } catch (error) {
        console.error('Health check failed:', error);
        showNotification('Health check failed - server may be down', 'error');
    }
}

function saveFormState() {
    const formState = {
        phaseType: elements.phaseType.value,
        phaseNum: elements.phaseNum.value,
        numSamples: elements.numSamples.value,
        plotType: elements.plotType.value
    };
    
    try {
        localStorage.setItem('tsSegmentationFormState', JSON.stringify(formState));
    } catch (error) {
        console.warn('Could not save form state:', error);
    }
}

function loadFormState() {
    try {
        const savedState = localStorage.getItem('tsSegmentationFormState');
        if (savedState) {
            const formState = JSON.parse(savedState);
            
            if (formState.phaseType) elements.phaseType.value = formState.phaseType;
            if (formState.phaseNum) elements.phaseNum.value = formState.phaseNum;
            if (formState.numSamples) elements.numSamples.value = formState.numSamples;
            if (formState.plotType) elements.plotType.value = formState.plotType;
        }
    } catch (error) {
        console.warn('Could not load form state:', error);
    }
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas ${getNotificationIcon(type)}"></i>
            <span>${message}</span>
        </div>
        <button class="notification-close" onclick="closeNotification(this)">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => notification.classList.add('show'), 100);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        closeNotification(notification.querySelector('.notification-close'));
    }, 5000);
}

function getNotificationIcon(type) {
    switch (type) {
        case 'success': return 'fa-check-circle';
        case 'error': return 'fa-exclamation-circle';
        case 'warning': return 'fa-exclamation-triangle';
        default: return 'fa-info-circle';
    }
}

function closeNotification(button) {
    const notification = button.closest('.notification');
    notification.classList.remove('show');
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 300);
}

function toggleFullscreen() {
    const plotContainer = document.querySelector('.plot-container');
    
    if (!document.fullscreenElement) {
        plotContainer.requestFullscreen().catch(err => {
            console.error('Error attempting to enable fullscreen:', err);
            showNotification('Fullscreen not supported', 'warning');
        });
    } else {
        document.exitFullscreen();
    }
}

function toggleMetrics() {
    const metricsContent = document.querySelector('.metrics-content');
    const toggleButton = document.querySelector('#metricsPanel .btn-ghost i');
    
    if (metricsContent.style.display === 'none') {
        metricsContent.style.display = 'block';
        toggleButton.className = 'fas fa-chevron-down';
    } else {
        metricsContent.style.display = 'none';
        toggleButton.className = 'fas fa-chevron-right';
    }
}

function closeAllModals() {
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
        modal.style.display = 'none';
    });
}

function showHelp() {
    document.getElementById('helpModal').style.display = 'block';
}

function showAbout() {
    showNotification('Time Series Segmentation Visualization Tool v1.0.0', 'info');
}

function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
}

// Add click outside modal to close
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('modal')) {
        e.target.style.display = 'none';
    }
});

// Handle window resize for responsive behavior
window.addEventListener('resize', function() {
    // Add any responsive adjustments here if needed
    console.log('Window resized');
});

// Add error handling for uncaught errors
window.addEventListener('error', function(e) {
    console.error('Uncaught error:', e.error);
    showNotification('An unexpected error occurred', 'error');
});

// Add handling for unhandled promise rejections
window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    showNotification('An unexpected error occurred', 'error');
});

console.log('Time Series Segmentation Visualization App initialized');