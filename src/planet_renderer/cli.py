"""稳定行星渲染器入口。"""

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Taichi 单行星实时渲染器")
    parser.parse_args(argv)
    print("行星渲染入口正在 M0 中重建；参考大气原型请显式运行：")
    print("python -m experiments.atmosphere_reference.cli --preset day --backend auto --preview")
    return 0
