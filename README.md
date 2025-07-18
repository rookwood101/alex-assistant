1. Install librespot - on raspberry pi use https://github.com/dtcooper/raspotify
2. Install uv (astral.sh)
3. Run `librespot --name Alex Assistant --enable-oauth --system-cache .librespot-cache`
4. Open the link and then copy the url you are redirected to and curl it on the raspberry pi in a separate terminal session
uv run main.py