from pathlib import Path

from cleaner import process
from config import INPUT, OUTPUT

Path(OUTPUT).mkdir(parents=True, exist_ok=True)

files = sorted(Path(INPUT).glob("*.mp3"))

if not files:
    print(f"No MP3 files found in {INPUT}")
else:
    for file in files:
        print(f"Processing {file}")
        process(str(file))

print("Done")
