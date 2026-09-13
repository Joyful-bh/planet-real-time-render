"""兼容无需安装、直接运行的项目入口。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from planet_renderer.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
