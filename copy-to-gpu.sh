rsync -rv --delete --exclude='.git' --filter=':- .gitignore' ./ test-gpu-02:/workspace/latencyai
