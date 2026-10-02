#!/bin/bash
#  macOS twin of "Start Visualizer.bat". Double-click it in Finder.
#
#  Serving the folder over http is what makes the midi/aligned, energy/ and
#  presets/ dropdowns work (they read the server's directory listing), and
#  http://localhost:8000/ is also the URL an OBS Browser source needs.
#
#  THIS TERMINAL WINDOW IS THE SERVER. Leave it open while you work; closing
#  it stops the server. Run it again any time — if it is already serving it
#  just reopens the browser instead of failing on a busy port.
#
#  First time only, if Finder refuses to run it:
#      chmod +x start-visualizer.command
#  A .command file is what macOS will launch on a double-click; a plain .sh
#  opens in a text editor instead.

PORT=8000
URL="http://localhost:${PORT}/"

# Finder launches this from the user's home directory, not from the folder the
# file lives in, so every relative path would miss. BASH_SOURCE is the script
# itself regardless of where it was started from.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

# python3 only. macOS has not shipped a `python` since Catalina, and on the
# machines that still have one it is Python 2, which has no http.server.
if ! command -v python3 >/dev/null 2>&1; then
  echo
  echo "  python3 was not found."
  echo "  Install it from python.org, or run:  xcode-select --install"
  echo
  read -r -p "  Press return to close. "
  exit 1
fi

# lsof rather than netstat: macOS netstat has no -p and reports differently
if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo
  echo "  Already serving on port ${PORT} — opening the browser."
  echo "  The server is the other Terminal window; close that one to stop it."
  echo
  open "${URL}"
  sleep 2
  exit 0
fi

# Wait for the port to actually answer before opening the browser, or the
# browser races the server and lands on "can't connect".
(
  for _ in $(seq 1 40); do
    if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
      open "${URL}"
      exit 0
    fi
    sleep 0.25
  done
) &

echo
echo "  Serving  $(pwd)"
echo "  Open at  ${URL}"
echo
echo "  Keep this window open. Closing it stops the server."
echo

# Bound to localhost on purpose: nothing outside this machine needs to reach
# it, and it stops macOS asking to accept incoming connections every launch.
python3 -m http.server "${PORT}" --bind 127.0.0.1

echo
echo "  Server stopped. If that was not on purpose, the reason is above —"
echo "  usually another program is already using port ${PORT}."
echo
read -r -p "  Press return to close. "
