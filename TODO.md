* Ensure system prompt includes
    - Upcoming calendar events for this week
    - Weather for today and next 7 days in home town
    - Current date/time
    - Current location
* Add a tool that allows it to remember things (forever, or for a particular date), which then get added to system prompt

* Tool for adding calendar event
* Echo cancellation doesn't work well enough - need to only listen for wakeword while gemini is talking

* Audio I/O
  - Autodetect mic input (prefer `seeed2micvoicec` / `seeed2mic`) in real mode; log devices and selection
  - Autodetect speaker output (prefer `UACDemoV1.0`) and handle stereo output cleanly; log selection
  - Keep overrides via `ALEX_INPUT_DEVICE_{INDEX,NAME}` and `ALEX_OUTPUT_DEVICE_{INDEX,NAME}`