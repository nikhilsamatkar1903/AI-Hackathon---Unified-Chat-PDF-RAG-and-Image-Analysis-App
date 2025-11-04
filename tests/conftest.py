import sys
import os

# Make src/app available for imports as "components" and main
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
APP_SRC = os.path.join(ROOT, 'src', 'app')
if APP_SRC not in sys.path:
    sys.path.insert(0, APP_SRC)
