#!/usr/bin/env python3
"""
Extract SYSTEM_PROMPT from bot.py at each git commit and save to files.
"""
import os
import re
import subprocess
from pathlib import Path

# Create output directory
output_dir = Path(os.path.expanduser("~/SpaceTraders/system-prompts"))
output_dir.mkdir(exist_ok=True)

# Get current commit to restore later
current_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd="/home/sic/SpaceTraders", text=True
).strip()

# Get all commits
commits = (
    subprocess.check_output(
        ["git", "log", "--oneline", "--reverse"],
        cwd="/home/sic/SpaceTraders",
        text=True,
    )
    .strip()
    .split("\n")
)

print(f"Found {len(commits)} commits")
print(f"Saving to: {output_dir}")

for i, line in enumerate(commits):
    short_hash, message = line.split(" ", 1)
    print(f"[{i+1}/{len(commits)}] {short_hash}: {message[:50]}...", end=" ")

    try:
        # Checkout commit
        subprocess.run(
            ["git", "checkout", short_hash],
            cwd="/home/sic/SpaceTraders",
            capture_output=True,
            check=True,
        )

        # Read bot.py
        bot_path = Path("/home/sic/SpaceTraders/bot.py")
        if not bot_path.exists():
            print("❌ bot.py not found")
            continue

        content = bot_path.read_text()

        # Extract SYSTEM_PROMPT using regex
        match = re.search(r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""', content, re.DOTALL)
        if not match:
            print("❌ SYSTEM_PROMPT not found")
            continue

        prompt = match.group(1)

        # Save to file
        output_file = output_dir / f"{short_hash}.txt"
        output_file.write_text(prompt)
        print(f"✓ {len(prompt)} chars")

    except subprocess.CalledProcessError as e:
        print(f"❌ git error: {e}")
    except Exception as e:
        print(f"❌ error: {e}")

# Restore original commit
print(f"\nRestoring to: {current_commit}")
subprocess.run(
    ["git", "checkout", current_commit],
    cwd="/home/sic/SpaceTraders",
    capture_output=True,
)

print(f"✓ Done! System prompts saved to {output_dir}")
print(f"View with: ls -la {output_dir}")
print(f"Compare: diff {output_dir}/commit1.txt {output_dir}/commit2.txt")
