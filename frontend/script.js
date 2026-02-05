// ============================================
// CONFIG & STATE
// ============================================
const API_URL = "http://localhost:8000";
let uploadedFile = null;
let resultImageUrl = null;
let resultMaskUrl = null;

// ============================================
// DOM ELEMENTS
// ============================================
const uploadArea = document.getElementById("uploadArea");
const imageInput = document.getElementById("imageInput");
const imagePreviewContainer = document.getElementById("imagePreviewContainer");
const imagePreview = document.getElementById("imagePreview");
const promptSelect = document.getElementById("promptSelect");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const loadingMsg = document.getElementById("loadingMsg");
const successMsg = document.getElementById("successMsg");
const errorMsg = document.getElementById("errorMsg");
const infoMsg = document.getElementById("infoMsg");
const resultsSection = document.getElementById("resultsSection");
const placeholderMsg = document.getElementById("placeholderMsg");
const statsBox = document.getElementById("statsBox");
const comparisonSlider = document.getElementById("comparisonSlider");
const sliderHandle = document.getElementById("sliderHandle");
const sliderBefore = document.getElementById("sliderBefore");
const sliderAfter = document.getElementById("sliderAfter");

// ============================================
// EVENT LISTENERS
// ============================================

// Upload area interactions
uploadArea.addEventListener("click", () => imageInput.click());

uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragging");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragging");
});

uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragging");
    handleFileUpload(e.dataTransfer.files[0]);
});

imageInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

// Comparison slider
let isSliding = false;
sliderHandle.addEventListener("mousedown", () => {
    isSliding = true;
});
document.addEventListener("mouseup", () => {
    isSliding = false;
});
document.addEventListener("mousemove", (e) => {
    if (!isSliding || !comparisonSlider.classList.contains("hidden")) return;
    
    const rect = comparisonSlider.getBoundingClientRect();
    let x = e.clientX - rect.left;
    
    if (x < 0) x = 0;
    if (x > rect.width) x = rect.width;
    
    const percent = (x / rect.width) * 100;
    sliderAfter.style.width = percent + "%";
    sliderHandle.style.left = percent + "%";
});

// ============================================
// MAIN FUNCTIONS
// ============================================

function handleFileUpload(file) {
    if (!file.type.startsWith("image/")) {
        showError("Please upload a valid image file");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showError("File size must be less than 10MB");
        return;
    }

    uploadedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreviewContainer.classList.remove("hidden");
        uploadArea.style.opacity = "0.6";
        infoMsg.classList.remove("active");
        
        // Add animation
        imagePreviewContainer.style.animation = "slideIn 0.5s ease-out";
    };
    reader.readAsDataURL(file);
}

async function predictSegmentation() {
    if (!uploadedFile) {
        showError("Please upload an image first");
        return;
    }

    const prompt = promptSelect.value;
    if (!prompt) {
        showError("Please select a detection type");
        return;
    }

    predictBtn.disabled = true;
    loadingMsg.classList.add("active");
    clearMessages();

    const startTime = performance.now();

    try {
        const formData = new FormData();
        formData.append("image", uploadedFile);
        formData.append("prompt", prompt);

        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            showError(error.detail || "Inference failed");
            return;
        }

        const maskBlob = await response.blob();
        resultMaskUrl = URL.createObjectURL(maskBlob);
        resultImageUrl = imagePreview.src;

        const endTime = performance.now();
        const inferenceTime = (endTime - startTime).toFixed(1);

        // Display results
        document.getElementById("resultImage").src = resultImageUrl;
        document.getElementById("resultMask").src = resultMaskUrl;
        
        // Setup comparison slider
        sliderBefore.src = resultImageUrl;
        sliderAfter.src = resultMaskUrl;
        
        resultsSection.classList.remove("hidden");
        comparisonSlider.classList.remove("hidden");
        placeholderMsg.classList.add("hidden");

        // Update stats
        updateStats(inferenceTime);
        statsBox.classList.remove("hidden");

        showSuccess(`✨ Segmentation complete in ${inferenceTime}ms!`);

    } catch (error) {
        console.error("Error:", error);
        showError(`Network error: ${error.message}`);
    } finally {
        predictBtn.disabled = false;
        loadingMsg.classList.remove("active");
    }
}

function clearAll() {
    uploadedFile = null;
    resultImageUrl = null;
    resultMaskUrl = null;
    imageInput.value = "";
    promptSelect.value = "";
    imagePreviewContainer.classList.add("hidden");
    uploadArea.style.opacity = "1";
    resultsSection.classList.add("hidden");
    comparisonSlider.classList.add("hidden");
    placeholderMsg.classList.remove("hidden");
    statsBox.classList.add("hidden");
    clearMessages();
    infoMsg.classList.add("active");
}

function updateStats(inferenceTime) {
    // Processing time
    document.getElementById("inferenceTime").textContent = inferenceTime + " ms";
    
    // File size
    const sizeKB = (uploadedFile.size / 1024).toFixed(1);
    document.getElementById("imageSize").textContent = sizeKB + " KB";
    
    // Resolution estimate
    document.getElementById("maskPixels").textContent = "256×256";
}

function showSuccess(message) {
    successMsg.textContent = message;
    successMsg.classList.add("active");
    errorMsg.classList.remove("active");
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        successMsg.classList.remove("active");
    }, 5000);
}

function showError(message) {
    errorMsg.textContent = "❌ " + message;
    errorMsg.classList.add("active");
    successMsg.classList.remove("active");
}

function clearMessages() {
    successMsg.classList.remove("active");
    errorMsg.classList.remove("active");
}

// ============================================
// INITIALIZATION
// ============================================
console.log("🚀 AI Segmentation Studio - Ready!");
console.log(`📡 API: ${API_URL}`);
console.log("🎨 Powered by ResNet18 + UNet");
