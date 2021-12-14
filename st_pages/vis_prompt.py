"""Colorized output text"""


class SysColors:
    OK = '\033[92m'  # GREEN
    WARNING = '\033[93m'  # YELLOW
    FAIL = '\033[91m'  # RED
    INFO = '\033[90m'  # GREY
    STRIKE = '\033[9m'  # strikethrough
    BOLD = '\033[1m'  # bold
    ITALIC = '\033[3m'  # italic
    UNDERLINED = '\033[4m'  # underlined
    BACK = '\033[7m'  # background
    RANDOM = '\033[95m'  # RANDOM COLOR
    RESET = '\033[0m'  # RESET COLOR


if __name__ == "__main__":
    print(SysColors.OK + "File Saved Successfully!" + SysColors.RESET)
    print(SysColors.WARNING + "Warning: Are you sure you want to continue?" + SysColors.RESET)
    print(SysColors.FAIL + "Unable to delete DataFrame." + SysColors.RESET)
    print(SysColors.RANDOM + "Let\'s try random color now." + SysColors.RESET)
