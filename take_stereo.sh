#!/bin/bash
echo "=== STEREO PHOTO CAPTURE ==="
echo "Position camera at robot height, pointing at scene"
echo ""
echo "Press ENTER to take Photo A..."
read
termux-camera-photo ~/robot/test_photos/stereo_a.jpg
echo "Photo A taken!"
echo ""
echo "Now move 5cm to the RIGHT (same height, same angle)"
echo "Press ENTER to take Photo B..."
read
termux-camera-photo ~/robot/test_photos/stereo_b.jpg
echo "Photo B taken!"
echo ""
echo "Running depth analysis..."
python3 ~/robot/stereo_depth.py ~/robot/test_photos/stereo_a.jpg ~/robot/test_photos/stereo_b.jpg 5.0 2>/dev/null
