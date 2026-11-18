import spaces
import sys
from pathlib import Path

# Add 'src' directory to sys.path so 'spokesman' package can be discovered
# both locally and in Hugging Face Spaces without requiring package installation.
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from spokesman.main import spokesman

if __name__ == "__main__":
    spokesman()
