"""
用法:
python3 print_cols.py <csv文件路径>
  打印所有列的编号和列名

python3 print_cols.py <csv文件路径> sort <列编号> <asc|desc>
  按指定列编号升序或降序排序，并生成新文件

生成文件名:
<原文件名>_sorted_by_col_<列编号>_<asc|desc>.csv
"""

import csv
import sys
from pathlib import Path


def usage() -> None:
    print(f"用法: python {sys.argv[0]} <csv文件路径>")
    print("      打印所有列的编号和列名")
    print(f"      python {sys.argv[0]} <csv文件路径> sort <列编号> <asc|desc>")
    print("      按指定列编号升序或降序排序，并生成新文件")
    print("      新文件名: <原文件名>_sorted_by_col_<列编号>_<asc|desc>.csv")


def rows(csv_file: str):
    return csv.DictReader(open(csv_file, newline="", encoding="utf-8-sig"))


def headers(csv_file: str):
    return rows(csv_file).fieldnames or []


def print_cols(csv_file: str) -> None:
    for i, name in enumerate(headers(csv_file)):
        print(f"{i}\t{name}")


def num(x: str) -> float:
    try:
        return float((x or "").strip().replace("\t", ""))
    except ValueError:
        return float("-inf")


def sort_csv(csv_file: str, col: int, desc: bool) -> None:
    reader = rows(csv_file)
    names = reader.fieldnames or []
    if col < 0 or col >= len(names):
        raise SystemExit(f"列编号超出范围: {col}")
    key = names[col]
    data = sorted(reader, key=lambda r: num(r.get(key, "")), reverse=desc)
    src = Path(csv_file)
    dst = src.with_name(f"{src.stem}_sorted_by_col_{col}_{'desc' if desc else 'asc'}{src.suffix}")
    with open(dst, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(data)
    print(f"Sorted file writed in {dst}")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in {"-h", "--help"}:
        usage()
    elif len(sys.argv) == 2:
        print_cols(sys.argv[1])
    elif len(sys.argv) == 5 and sys.argv[2] == "sort" and sys.argv[4] in {"asc", "desc"}:
        sort_csv(sys.argv[1], int(sys.argv[3]), sys.argv[4] == "desc")
    else:
        usage()
        sys.exit(1)
