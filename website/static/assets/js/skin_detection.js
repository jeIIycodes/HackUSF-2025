const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const overlay = document.getElementById('overlay');
const overlayCtx = overlay.getContext('2d');
const statusBox = document.getElementById('detection-status');
const aiResult = document.getElementById('ai-result');

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  })
  .catch(err => {
    statusBox.innerText = '🚫 Unable to access camera';
    statusBox.className = 'text-center fw-bold mb-4 text-danger';
    console.error(err);
  });

// Every 2 seconds, capture frame and send to Flask backend
setInterval(() => {
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Convert to base64
  const dataURL = canvas.toDataURL('image/jpeg');
  const blob = dataURItoBlob(dataURL);
  const formData = new FormData();
  formData.append('image', blob, 'frame.jpg');

  fetch('http://localhost:5001/analyze', {
    method: 'POST',
    body: formData
  })
  .then(res => res.json())
  .then(data => {
    if (data.error) {
      aiResult.innerText = `⚠️ Error: ${data.error}`;
      aiResult.className = 'text-center fw-bold mb-4 text-danger';
      return;
    }

    const label = data.classification;
    const prob = (data.probability * 100).toFixed(2);

    statusBox.innerText = '✅ Frame analyzed!';
    statusBox.className = 'text-center fw-bold mb-4 text-success';

    aiResult.innerHTML = `
      <div class="alert ${label === 'Malignant' ? 'alert-danger' : 'alert-success'}">
        <h5>🧬 Prediction: <strong>${label}</strong></h5>
        <p>Confidence: ${prob}%</p>
      </div>
    `;
  })
  .catch(err => {
    aiResult.innerText = `❌ Failed to analyze: ${err}`;
    aiResult.className = 'text-center fw-bold mb-4 text-danger';
    console.error(err);
  });

}, 2000); // every 2 seconds

// Helper: convert base64 to Blob
function dataURItoBlob(dataURI) {
  const byteString = atob(dataURI.split(',')[1]);
  const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
  const ab = new ArrayBuffer(byteString.length);
  const ia = new Uint8Array(ab);

  for (let i = 0; i < byteString.length; i++) {
    ia[i] = byteString.charCodeAt(i);
  }

  return new Blob([ab], { type: mimeString });
}
