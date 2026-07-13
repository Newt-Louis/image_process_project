const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const uploadBtn = document.getElementById('upload-btn');
const preview = document.getElementById('preview');
const detectBtn = document.getElementById('detect-btn');
const loader = document.getElementById('loader');
const results = document.getElementById('results');
const annotatedImage = document.getElementById('annotated-image');
const cropsContainer = document.getElementById('crops-container');

let currentFile = null;

// Trigger file input
uploadBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});
dropArea.addEventListener('click', () => fileInput.click());

// Drag and drop handling
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
});

dropArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

fileInput.addEventListener('change', function() {
    handleFiles(this.files);
});

function handleFiles(files) {
    if (files.length > 0) {
        currentFile = files[0];
        
        // Ensure it's an image
        if (!currentFile.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }

        const reader = new FileReader();
        reader.readAsDataURL(currentFile);
        reader.onloadend = function() {
            preview.src = reader.result;
            preview.style.display = 'block';
            
            // Hide text and button inside drop area for cleaner look
            uploadBtn.style.display = 'none';
            dropArea.querySelector('p').style.display = 'none';
            
            // Enable detect button
            detectBtn.classList.remove('disabled');
            detectBtn.disabled = false;
            
            // Hide old results
            results.style.display = 'none';
        }
    }
}

detectBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // Show loading
    detectBtn.classList.add('disabled');
    detectBtn.disabled = true;
    loader.style.display = 'block';
    results.style.display = 'none';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('model_name', document.getElementById('model-select').value);

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const data = await response.json();
        
        // Display annotated image
        annotatedImage.src = data.annotated_image;
        
        // Display crops
        cropsContainer.innerHTML = '';
        if (data.crops && data.crops.length > 0) {
            data.crops.forEach(crop => {
                const cropDiv = document.createElement('div');
                cropDiv.className = 'crop-item';
                cropDiv.innerHTML = `
                    <img src="${crop.image}" alt="Cropped product">
                    <span>${crop.label}</span>
                `;
                cropsContainer.appendChild(cropDiv);
            });
        } else {
            cropsContainer.innerHTML = '<p style="color: var(--text-muted)">No objects detected.</p>';
        }

        // Show results
        results.style.display = 'flex';

    } catch (error) {
        alert('Error during detection: ' + error.message);
    } finally {
        loader.style.display = 'none';
        detectBtn.classList.remove('disabled');
        detectBtn.disabled = false;
    }
});
