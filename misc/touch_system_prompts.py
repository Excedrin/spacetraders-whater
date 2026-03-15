#!/usr/bin/env python3
"""
Set modification time of system prompt files to match commit timestamps.
"""
import os
import subprocess
from datetime import datetime
from pathlib import Path

output_dir = Path(os.path.expanduser("~/SpaceTraders/system-prompts"))

# Get all commits with timestamps
commits = (
    subprocess.check_output(
        ["git", "log", "--oneline", "--format=%h %cI"],
        cwd="/home/sic/SpaceTraders",
        text=True,
    )
    .strip()
    .split("\n")
)

print(f"Setting timestamps for {len(commits)} files...")

for line in commits:
    parts = line.split(" ", 1)
    if len(parts) < 2:
        continue

    short_hash = parts[0]
    timestamp = parts[1]

    file_path = output_dir / f"{short_hash}.txt"

    if not file_path.exists():
        print(f"❌ {short_hash}: file not found")
        continue

    try:
        # Parse ISO timestamp and convert to epoch
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        epoch = int(dt.timestamp())

        # Set file modification time
        os.utime(file_path, (epoch, epoch))

        # Show human-readable date
        date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        print(f"✓ {short_hash}: {date_str}")
    except Exception as e:
        print(f"❌ {short_hash}: {e}")

print(f"\n✓ Done! Files now have commit timestamps.")
print(f"\nView timeline with:")
print(f"  ls -lh --full-time {output_dir} | tail -20")
print(f"  ls -lhtr {output_dir}  # Oldest first")
