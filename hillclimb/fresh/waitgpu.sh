# exit when no llama-server/bench/perplexity process for 3 consecutive checks 40 s apart
n=0
while [ $n -lt 3 ]; do
  if powershell -NoProfile -c "if (Get-Process llama-server,llama-bench,llama-perplexity -ErrorAction SilentlyContinue) {exit 1} else {exit 0}"; then n=$((n+1)); else n=0; fi
  sleep 40
done
echo "GPU free $(date)"
