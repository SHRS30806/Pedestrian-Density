# Research Demonstration Quickstart 🚀

This guide provides instructions for academic peers and reviewers to locally reproduce the live inference dashboard. By following these steps, you can execute the pre-trained PPO model and visually validate the real-time "Sim-to-Real" transfer capabilities using global CCTV internet streams.

## Prerequisites
Ensure you have Python 3.8+ installed on your computer.

---

## Step 1: Install Dependencies
Open your terminal (or command prompt), navigate to the `python` folder of this project, and install the required libraries:

```bash
cd python
pip install -r requirements.txt
```
*(If you don't have a requirements file, run: `pip install yt-dlp flask flask-cors ultralytics opencv-python torch numpy`)*

## Step 2: Start the Backend Server
The Flask server is required to run the PyTorch model and the YOLOv8 object detection. It will process the video feeds and stream them to your browser.

Run this command inside the `python` directory:
```bash
python server.py
```
*You should see output indicating the server is running on `http://127.0.0.1:5000`.*

## Step 3: Open the Dashboard
Keep the terminal open (the server needs to stay running).

Open your favorite web browser (Chrome, Edge, Firefox) and navigate to:
👉 **http://127.0.0.1:5000**

### Verifying the Inference:
1. **Simulation Verification:** By default, the interface will display the mathematical MDP bounds and the 24D State Vector updating in real-time.
2. **Live Inference Validation:** Select the **"Global Map"** to trigger the Flask backend to extract an `.m3u8` internet stream. This allows you to observe the YOLOv8 and PPO pipeline operating on unscripted, real-world traffic intersections.
3. **Local Dataset Testing:** Use the **"Upload Real Video"** module to test the pipeline against any proprietary `.mp4` traffic dataset.
4. **Predictive Radar Mock-up:** The UI includes a secondary component simulating future GPS telemetry fusion for incoming platoon detection.
