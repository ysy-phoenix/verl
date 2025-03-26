import ast
import warnings

BLOCK_LIBS = [
    "fake-useragent",
    "keras",
    "socket",
    "torch",
    "scipy",
    "sklearn",
    "cv2",
    "scipy",
    "imageio",
    "sphinx-pyproject",
    "xgboost",
    "tweepy",
    "flask",
    "matplotlib",
    "pillow",
    "seaborn",
    "smtpd",
]

warnings.filterwarnings("ignore", category=SyntaxWarning)


def check_code_with_ast(code):
    try:
        ast.parse(code)
        compile(code, "<string>", "exec")
        if "eval(" in code or "exec(" in code:  # Filter out unsafe code
            return False
        for lib in BLOCK_LIBS:  # Filter out unsafe libraries
            if lib in code:
                return False
        return True
    except SyntaxError:
        return False
