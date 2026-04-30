cd python
echo "--- Running Experiment (Fast Mode) ---"
..\venv\Scripts\python.exe run_experiment.py --fast

echo "--- Updating Git ---"
cd ..
git add .
git commit -m "fix: replace unicode characters, update positive reward plots"
git push origin main
