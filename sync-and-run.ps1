# PowerShell script to sync using WSL rsync and run on Raspberry Pi

# Config
$piHost = "pi@raspberrypi.local"
$remotePath = "/home/pi/alex-assistant"
$localPath = "."  # current dir

# Execute via WSL
git add .
git commit -m "sync"
git push
ssh $piHost "cd $remotePath && git pull && /home/pi/.local/bin/uv run main.py"