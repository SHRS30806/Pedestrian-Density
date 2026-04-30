cd python
echo "--- Running Training ---"
..\venv\Scripts\python.exe train.py --cfg configs/default.yaml

echo "--- Running Evaluation ---"
..\venv\Scripts\python.exe run_experiment.py

echo "--- Running Plotting ---"
..\venv\Scripts\python.exe plot_results.py

echo "--- Updating Git ---"
cd ..
$gitignore = Get-Content .gitignore -Raw
$gitignore = $gitignore -replace "results/`r`n", ""
$gitignore = $gitignore -replace "python/results/`r`n", ""
Set-Content .gitignore $gitignore

git add .
git commit -m "chore: generate trained model, evaluation metrics, and plots"
git push origin main
