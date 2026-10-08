"""2.4 节文献口径普查的取证工具：把论文 PDF 抽成文本，再按关键词定位原句。

为什么要有这个脚本
------------------
这份普查的全部价值在于「**真读正文**」。如果判定依据是某个摘要工具的转述，
那张表就只是二手印象，审稿人一查就塌。所以流程定成：

  下载 PDF -> 本地抽文本 -> 按关键词把**原句**打出来 -> 人眼判定 -> 记进
  docs/protocol-audit.md（连原句一起记）

脚本只做「找出原句」这一步，**不做判定**。判定是人的事，
因为「报告了轮数」和「报告了收敛判据」的区别需要读上下文才能分清。

🔴 一个必须守住的限制
--------------------
pypdf 抽出来的文本有三类已知失真，判定时要当心：
  1. 两栏排版常被抽成交错的行，句子会断在奇怪的地方；
  2. 连字符断词（如 "train-\ning"）会留下断口；
  3. 表格会塌成一串数字，失去行列对应关系。
所以本脚本打印的是**命中词前后各一段**，而不是单句——
上下文不够时必须回去翻原 PDF 的那一页，不要凭残句下判断。
命中数为 0 **不等于**论文没报告：也可能是这一页是扫描图、或抽取失败。
故脚本会同时打印每页抽出的字符数，页面大面积为 0 就说明抽取不可信。

用法
----
    # 下载并抽取（存到 refs/ 下，PDF 不入库）
    PYTHONIOENCODING=utf-8 python -m src.audit_fetch get <键名> <URL>

    # 对已下载的论文按默认关键词组检索
    PYTHONIOENCODING=utf-8 python -m src.audit_fetch scan <键名>

    # 自定义关键词
    PYTHONIOENCODING=utf-8 python -m src.audit_fetch scan <键名> --kw epoch iteration

    # 列出已下载的
    PYTHONIOENCODING=utf-8 python -m src.audit_fetch list
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

REF_DIR = Path("refs")          # PDF 与抽出的 txt 都放这里，整个目录不入库
UA = "Mozilla/5.0 (compatible; academic-reading-script)"

# 默认关键词组：覆盖本普查要核的三项 + 两项附加项。
# 分组是为了让输出有结构，便于逐项填表。
KEYWORD_GROUPS = {
    "①轮数": ["epoch"],
    "②总步数": ["iteration", "step", "update", "minibatch", "mini-batch"],
    "③收敛/早停": ["converge", "convergence", "early stop", "early-stop", "patience",
                "plateau", "best checkpoint", "best model", "until no", "stopping criterion"],
    "批大小": ["batch size", "batch-size"],
    "预算曲线": ["budget", "labeled set", "labelled set", "annotation cost",
             "fraction of", "percentage of", "% of the", "labeled data", "labelled data"],
    "重训与否": ["from scratch", "re-train", "retrain", "fine-tune", "finetune",
             "warm start", "warm-start"],
}


def pdf_path(key: str) -> Path:
    return REF_DIR / f"{key}.pdf"


def txt_path(key: str) -> Path:
    return REF_DIR / f"{key}.txt"


def cmd_get(key: str, url: str) -> None:
    """下载 PDF 并抽成文本。两者都留档，便于事后复核。"""
    REF_DIR.mkdir(exist_ok=True)
    fp = pdf_path(key)
    if not fp.exists():
        print(f"下载 {url}")
        # 🔴 必须分块读，不能一次 r.read()。
        # 大 PDF（本普查遇到过 29 MB 的）一次性读会在中途抛 IncompleteRead，
        # 看起来像「下不到全文」，其实是读法的问题。把这种失败误记成
        # 「排除：无法获取全文」会让普查的分母凭空变小，所以这里必须读干净。
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        chunks, got = [], 0
        with urllib.request.urlopen(req, timeout=180) as r:
            expect = int(r.headers.get("Content-Length") or 0)
            while True:
                try:
                    buf = r.read(1 << 16)
                except Exception as e:      # 连接中断：保留已读到的部分再判断
                    print(f"  ⚠️ 读取中断（{e}），已读 {got/1024:.0f} KB")
                    break
                if not buf:
                    break
                chunks.append(buf)
                got += len(buf)
        data = b"".join(chunks)
        if expect and got < expect:
            raise SystemExit(
                f"只读到 {got} / {expect} 字节，文件不完整。重试或换镜像；"
                f"**不要**按「无法获取全文」记录——那是下载问题，不是获取性问题。")
        if not data.startswith(b"%PDF"):
            raise SystemExit(
                f"拿到的不是 PDF（开头是 {data[:16]!r}）。多半是被挡在付费墙或人机验证外，"
                f"按规则这篇应记为「排除：无法获取全文」，不要换一篇顶上。")
        fp.write_bytes(data)
        print(f"已存 {fp}（{len(data)/1024:.0f} KB）")
    else:
        print(f"已存在 {fp}，跳过下载")

    from pypdf import PdfReader
    reader = PdfReader(str(fp))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception as e:                      # 个别页抽取失败不该中断整篇
            pages.append("")
            print(f"  ⚠️ 第 {i+1} 页抽取失败：{e}")

    # 🔴 抽取质量自检：抽不出字的页数太多，后面的「未命中」就不可信
    empty = [i + 1 for i, t in enumerate(pages) if len(t.strip()) < 50
             and i < len(pages) - 2]   # 末两页常是参考文献尾巴或空白，不计
    txt_path(key).write_text("\n\n".join(
        f"=== [page {i+1}] ===\n{t}" for i, t in enumerate(pages)), encoding="utf-8")
    total = sum(len(t) for t in pages)
    print(f"共 {len(pages)} 页，抽出 {total} 字符 -> {txt_path(key)}")
    if empty:
        print(f"  ⚠️ 以下页几乎没抽出文字，可能是图片页：{empty}")
        print(f"     这些页上的内容本脚本看不到，判定「未报告」前须人工翻查。")
    if total < 5000:
        print("  🔴 全篇抽出的字符过少，抽取很可能失败。"
              "此篇不得按「未命中」判定为未报告。")


def sentences_around(text: str, m: re.Match, width: int = 320) -> str:
    """取命中词前后一段，而不是单句——两栏 PDF 的断句不可靠。"""
    lo = max(0, m.start() - width)
    hi = min(len(text), m.end() + width)
    seg = text[lo:hi].replace("\n", " ")
    return re.sub(r"\s+", " ", seg).strip()


def cmd_scan(key: str, kws: list[str] | None, group: str | None) -> None:
    fp = txt_path(key)
    if not fp.exists():
        raise SystemExit(f"缺 {fp}，先跑 `get {key} <URL>`")
    text = fp.read_text(encoding="utf-8")

    groups = {"自定义": kws} if kws else KEYWORD_GROUPS
    if group:
        groups = {k: v for k, v in groups.items() if group in k}

    for gname, words in groups.items():
        print(f"\n{'='*78}\n{gname}  ({', '.join(words)})\n{'='*78}")
        hits = 0
        for w in words:
            for m in re.finditer(re.escape(w), text, re.I):
                # 页码：从命中位置往前找最近的页标记
                page = "?"
                pm = list(re.finditer(r"=== \[page (\d+)\] ===", text[:m.start()]))
                if pm:
                    page = pm[-1].group(1)
                print(f"\n  [p{page}] ...{sentences_around(text, m)}...")
                hits += 1
                if hits >= 14:          # 够判定即可，不要把整篇刷出来
                    print(f"\n  （该组命中已达 14 条，截断。需要更多请用 --kw 缩小关键词）")
                    break
            if hits >= 14:
                break
        if hits == 0:
            print("  —— 无命中。⚠️ 这不等于论文未报告，先确认抽取是否完整。")


def cmd_list() -> None:
    if not REF_DIR.exists():
        print("refs/ 还不存在")
        return
    for fp in sorted(REF_DIR.glob("*.pdf")):
        t = txt_path(fp.stem)
        n = len(t.read_text(encoding="utf-8")) if t.exists() else 0
        print(f"  {fp.stem:<28} pdf {fp.stat().st_size/1024:7.0f} KB   txt {n:>8} 字符")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("get");  g.add_argument("key"); g.add_argument("url")
    s = sub.add_parser("scan"); s.add_argument("key")
    s.add_argument("--kw", nargs="+", default=None)
    s.add_argument("--group", default=None)
    sub.add_parser("list")
    a = ap.parse_args()

    if a.cmd == "get":
        cmd_get(a.key, a.url)
    elif a.cmd == "scan":
        cmd_scan(a.key, a.kw, a.group)
    else:
        cmd_list()


if __name__ == "__main__":
    main()
