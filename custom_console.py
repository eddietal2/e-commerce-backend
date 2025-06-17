# Custom Module for Console Log formats,
# like Colors, Loading Indicators, etc
import sys
import time
import os

# ANSI escape codes for colors, etc, in print statements
COLOR_RED = "\033[91m"   # Bright Red
COLOR_GREEN = "\033[92m" # Bright Green
COLOR_YELLOW = "\033[93m" # Bright Yellow
COLOR_BLUE = "\033[94m"  # Bright Blue
COLOR_MAGENTA = "\033[95m"# Bright Magenta
COLOR_CYAN = "\033[96m"  # Bright Cyan
COLOR_WHITE = "\033[97m" # Bright White
RESET_COLOR = "\033[0m" # Reset to default color and formatting

def simple_spinner(duration=3):
    spinner_chars = ['-', '\\', '|', '/'] # Characters for the spinner
    start_time = time.time()
    i = 0
    while time.time() - start_time < duration:
        sys.stdout.write(f'\r{COLOR_MAGENTA}Loading {spinner_chars[i % len(spinner_chars)]}')
        sys.stdout.flush()
        time.sleep(0.1) # Controls the speed of the spin
        i += 1
    sys.stdout.write(f'{COLOR_YELLOW}\rLoading complete!') # Overwrite with final message and a newline
    sys.stdout.flush()

# Clear console log
def clear_console():
    """Clears the console screen based on the operating system."""
    os.system('cls' if os.name == 'nt' else 'clear')
