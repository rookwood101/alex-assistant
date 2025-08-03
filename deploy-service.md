# Deploy Alex Assistant Systemd Service

To deploy the systemd service on your Raspberry Pi:

1. Copy the service file to systemd directory:
```bash
sudo cp alex-assistant.service /etc/systemd/system/
```

2. Reload systemd to recognize the new service:
```bash
sudo systemctl daemon-reload
```

3. Enable the service to start on boot:
```bash
sudo systemctl enable alex-assistant.service
```

4. Start the service immediately:
```bash
sudo systemctl start alex-assistant.service
```

5. Check service status:
```bash
sudo systemctl status alex-assistant.service
```

6. View logs:
```bash
sudo journalctl -u alex-assistant.service -f
```

## Service Management Commands

- Stop service: `sudo systemctl stop alex-assistant.service`
- Restart service: `sudo systemctl restart alex-assistant.service`
- Disable auto-start: `sudo systemctl disable alex-assistant.service`