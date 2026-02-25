<h1 align="center">Cross-Image Person Re-Identification API</h1>

<p align="center">
A FastAPI-based backend service that detects persons in two images,
extracts identity embeddings, and matches individuals across images
using cosine similarity.
</p>

<hr>

<h2>Overview</h2>

<p>
This API performs cross-image person re-identification (ReID).
It detects persons in two input images, extracts deep identity embeddings,
and determines whether any individuals appear in both images.
</p>

<p>
The API returns:
</p>

<ul>
  <li>Total persons detected in each image</li>
  <li>Matched persons with similarity scores</li>
  <li>Unmatched person indices</li>
  <li>Bounding box coordinates</li>
</ul>

<hr>

<h2>Tech Stack</h2>

<ul>
  <li><b>FastAPI</b> – API framework</li>
  <li><b>YOLO</b> – Person detection</li>
  <li><b>OSNet (Torchreid)</b> – Person ReID embeddings</li>
  <li><b>NumPy</b> – Cosine similarity computation</li>
  <li><b>OpenCV</b> – Image processing</li>
  <li><b>PyTorch</b> – Deep learning backend</li>
</ul>

<hr>

<h2>API Endpoint</h2>

<h3>POST /compare</h3>

<p><b>Request:</b></p>

<ul>
  <li>image1 (JPEG or PNG)</li>
  <li>image2 (JPEG or PNG)</li>
</ul>

<p><b>Response Example:</b></p>

<pre>
{
  "image1_person_count": 3,
  "image2_person_count": 2,
  "matches": [
    {
      "image1_index": 0,
      "image2_index": 1,
      "similarity": 0.87,
      "bbox_image1": [100, 50, 300, 400],
      "bbox_image2": [120, 60, 320, 410]
    }
  ],
  "unmatched_image1": [1, 2],
  "unmatched_image2": [0]
}
</pre>

<hr>

<h2>How It Works</h2>

<ol>
  <li>Detect persons in both images using YOLO.</li>
  <li>Crop detected bounding boxes.</li>
  <li>Extract identity embeddings using OSNet.</li>
  <li>Compute cosine similarity between embeddings.</li>
  <li>Match individuals based on similarity threshold.</li>
</ol>

<hr>

<h2>Running Locally</h2>

<pre>
pip install -r requirements.txt
uvicorn app.main:app --reload --port 9000
</pre>

Access API docs at:

<pre>
http://127.0.0.1:9000/docs
</pre>

<hr>

<h2>Notes</h2>

<ul>
  <li>Supports JPEG and PNG formats.</li>
  <li>Images are resized if excessively large.</li>
  <li>Designed for demonstration and backend evaluation purposes.</li>
</ul>

<hr>

<p align="center">
Built for backend and computer vision evaluation.
</p>
