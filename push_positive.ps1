cd python
echo "--- Running Experiment (Fast Mode) ---"
..\venv\Scripts\python.exe run_experiment.py --fast

echo "--- Updating Git ---"
cd ..
$gitignore = Get-Content .gitignore -Raw
$gitignore = $gitignore -replace "results/`r`n", ""
$gitignore = $gitignore -replace "python/results/`r`n", ""
Set-Content .gitignore $gitignore

git add .
git commit -m "chore: generate trained model, evaluation metrics, and plots with positive reward format"
git push origin main
