# Scale Agent – Run on Google VM

1. Copy this folder to the VM (e.g. `~/scale_agent`) and put your `course_info.txt` inside it.
2. Setup:
   ```bash
   cd ~/scale_agent
   sudo apt update && sudo apt install -y python3-venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env && nano .env     # add GEMINI_API_KEY and DISCORD_BOT_TOKEN
   ```
3. Test run: `python bot.py`
4. Run 24/7 (survives logout/reboot):
   ```bash
   # edit YOUR_USER in scale-agent.service first
   sudo cp scale-agent.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now scale-agent
   journalctl -u scale-agent -f      # view logs
   ```
Note: in the Discord Developer Portal, enable "Message Content Intent" for the bot.
