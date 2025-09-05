let currentTrajectory = 0;
let totalTrajectories = 0;
let trajectoryData = [];

function initializeTrajectories() {
    totalTrajectories = trajectoryData.length;

    if (totalTrajectories > 0) {
        showTrajectory(0);
    }

    updateNavigation();
}

function buildTrajectoryHTML(index, row) {
    const conversation = row.conversation;
    const statusClass = `status-${row.status}`;

    let stepsHtml = '';
    for (let i = 0; i < conversation.length; i++) {
        const message = conversation[i];
        const role = message.role;
        const content = message.content;

        let bgClass = 'observation';
        if (role === 'user') {
            bgClass = 'action';
        } else if (role === 'assistant') {
            bgClass = 'response';
        }

        let stepHTML = '';

        // Display reasoning first if available (typically for assistant messages)
        if (message.reasoning) {
            stepHTML += `<div class="observation"><strong>Reasoning:</strong><br><pre>${message.reasoning}</pre></div>`;
        }

        // Then display the main message content
        stepHTML += `<div class="${bgClass}"><strong>${role.charAt(0).toUpperCase() + role.slice(1)}:</strong><br><pre>${content}</pre></div>`;

        if (message.step_metadata) {
            const metadata = message.step_metadata;
            let metadataHtml = '<div class="metadata"><strong>Step Metadata:</strong><br>';
            metadataHtml += `<div class="metadata-item"><strong>Score:</strong> ${metadata.score}</div>`;
            metadataHtml += `<div class="metadata-item"><strong>Moves:</strong> ${metadata.moves}</div>`;
            metadataHtml += `<div class="metadata-item"><strong>Location:</strong> ${metadata.location}</div>`;
            if (metadata.inventory && metadata.inventory.length > 0) {
                const inventoryArray = Array.isArray(metadata.inventory) ? metadata.inventory : [metadata.inventory];
                metadataHtml += `<div class="metadata-item"><strong>Inventory:</strong> ${inventoryArray.join(', ')}</div>`;
            }
            metadataHtml += '</div>';
            stepHTML += metadataHtml;
        }

        stepsHtml += `<div class="step">${stepHTML}</div>`;
    }

    return `
        <div class="trajectory-header">
            <span>
                Trajectory #${index + 1} - ${row.model} - Game: ${row.game}
                <span class="${statusClass}">[${row.status.toUpperCase()}]</span>
            </span>
        </div>
        <div class="trajectory-content">
            ${stepsHtml}
        </div>
    `;
}

function showTrajectory(index) {
    if (index < 0 || index >= totalTrajectories) return;

    const container = document.getElementById('trajectory-container');
    const row = trajectoryData[index];

    container.innerHTML = buildTrajectoryHTML(index, row);
    currentTrajectory = index;
    updateNavigation();
}

function nextTrajectory() {
    if (currentTrajectory < totalTrajectories - 1) {
        showTrajectory(currentTrajectory + 1);
    }
}

function previousTrajectory() {
    if (currentTrajectory > 0) {
        showTrajectory(currentTrajectory - 1);
    }
}

function updateNavigation() {
    const prevButton = document.getElementById('prev-button');
    const nextButton = document.getElementById('next-button');
    const trajectoryInfo = document.getElementById('trajectory-info');

    if (prevButton) prevButton.disabled = (currentTrajectory === 0);
    if (nextButton) nextButton.disabled = (currentTrajectory === totalTrajectories - 1);

    if (trajectoryInfo) {
        trajectoryInfo.textContent = `Trajectory ${currentTrajectory + 1} of ${totalTrajectories}`;
    }
}

// Function to set trajectory data from Python
function setTrajectoryData(data) {
    trajectoryData = data;
    initializeTrajectories();
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function () {
    // If trajectoryData is already set, initialize
    if (trajectoryData.length > 0) {
        initializeTrajectories();
    }
});
