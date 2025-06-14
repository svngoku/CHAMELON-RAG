import logging

# Color codes
COLORS = {
    'BLUE': '\033[94m',
    'GREEN': '\033[92m',
    'YELLOW': '\033[93m',
    'RED': '\033[91m',
    'CYAN': '\033[96m',
    'ENDC': '\033[0m'  # End color
}

def setup_colored_logger(level=logging.INFO):
    """Setup basic colored logging with configurable level."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger() 