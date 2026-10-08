"""从 RAR5 归档的**任意中间片段**里直接抽取文件 —— 不依赖 unrar。

为什么需要它
------------
RivAIrSet 打包成单个 101 GB 的 .rar，其中 2019 年的标签位于约 17.87 GB 偏移处。
按顺序下载到那里需要约 5 小时，但我们只要其中几百个 JSON（约几百 MB）。

Zenodo 支持 HTTP Range，可以直接取中间那一段。但 unrar 无法解析
「掐头」的归档 —— 它需要从归档头开始的完整结构。

解决办法：自己解析 RAR5 的文件头。RAR5 每个文件都自带完整头部
（不像 ZIP 依赖末尾的中央目录），所以只要在片段里找到头部起点，
就能定位数据区。**若压缩方式是 0（存储/未压缩），数据可直接读出。**

RAR5 文件头结构（vint = 小端 base-128 变长整数，高位为续标志）
--------------------------------------------------------------
    CRC32          4 字节
    HeadSize       vint   本字段之后到头部结束的字节数
    HeadType       vint   2 = 文件头
    HeadFlags      vint
    [ExtraSize]    vint   若 HeadFlags & 0x01
    [DataSize]     vint   若 HeadFlags & 0x02   <- 数据区字节数
    FileFlags      vint
    UnpackedSize   vint
    Attributes     vint
    [mtime]        4 字节 若 FileFlags & 0x02
    [DataCRC]      4 字节 若 FileFlags & 0x04
    CompInfo       vint   bit7-9 = 压缩方式，0 表示仅存储
    HostOS         vint
    NameLength     vint
    Name           NameLength 字节 (UTF-8)
    [ExtraArea]
    <数据区紧随其后>
"""
from __future__ import annotations

import re
from dataclasses import dataclass


def read_vint(buf: bytes, pos: int) -> tuple[int, int]:
    """读一个变长整数，返回 (值, 新位置)。"""
    value = 0
    shift = 0
    for _ in range(10):  # 最多 10 字节，防止损坏数据导致死循环
        if pos >= len(buf):
            raise ValueError("vint 越界")
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not (b & 0x80):
            return value, pos
        shift += 7
    raise ValueError("vint 过长")


@dataclass
class FileEntry:
    name: str
    data_offset: int      # 数据区在 buf 中的起始位置
    data_size: int        # 数据区字节数（压缩后）
    unpacked_size: int    # 原始大小
    method: int           # 压缩方式，0 = 仅存储
    header_offset: int    # 头部起点


def parse_file_header(buf: bytes, start: int) -> FileEntry | None:
    """尝试把 start 处当作一个文件头来解析。失败返回 None。"""
    try:
        pos = start + 4                      # 跳过 CRC32
        head_size, pos = read_vint(buf, pos)
        head_end = pos + head_size           # 头部结束位置 = 数据区起点

        head_type, pos = read_vint(buf, pos)
        if head_type != 2:                   # 只要文件头
            return None
        head_flags, pos = read_vint(buf, pos)

        if head_flags & 0x01:                # 有 extra area
            _, pos = read_vint(buf, pos)
        data_size = 0
        if head_flags & 0x02:                # 有数据区
            data_size, pos = read_vint(buf, pos)

        file_flags, pos = read_vint(buf, pos)
        unpacked_size, pos = read_vint(buf, pos)
        _attr, pos = read_vint(buf, pos)

        if file_flags & 0x02:                # mtime
            pos += 4
        if file_flags & 0x04:                # data CRC
            pos += 4

        comp_info, pos = read_vint(buf, pos)
        method = (comp_info >> 7) & 0x07     # bit7-9 = 压缩方式
        _host_os, pos = read_vint(buf, pos)
        name_len, pos = read_vint(buf, pos)

        if name_len <= 0 or name_len > 4096 or pos + name_len > len(buf):
            return None
        name = buf[pos : pos + name_len].decode("utf-8", errors="strict")

        # 合理性校验：名字必须是可见路径，且头部结束位置要在缓冲区内
        if not name or "\x00" in name:
            return None

        return FileEntry(
            name=name, data_offset=head_end, data_size=data_size,
            unpacked_size=unpacked_size, method=method, header_offset=start,
        )
    except (ValueError, UnicodeDecodeError, IndexError):
        return None


def find_entries(buf: bytes, name_pattern: bytes = rb"[A-Za-z0-9_./\\-]+\.(?:json|JPG)") -> list[FileEntry]:
    """在缓冲区里扫描所有能成功解析的文件头。

    做法：先用正则找出所有像文件名的位置，再从每个位置向前回退若干字节
    逐一尝试解析头部，命中即认为找到。
    比盲目逐字节扫描快几个数量级。
    """
    entries: list[FileEntry] = []
    seen: set[int] = set()

    for m in re.finditer(name_pattern, buf):
        name_off = m.start()
        # 文件名在头部靠后位置，头起点通常在其前 20~120 字节内
        for back in range(16, 140):
            start = name_off - back
            if start < 0 or start in seen:
                continue
            e = parse_file_header(buf, start)
            # 校验解析出的名字确实落在我们找到的位置上
            if e and e.name.encode("utf-8", "ignore").endswith(
                m.group(0).split(b"/")[-1].split(b"\\")[-1]
            ):
                entries.append(e)
                seen.add(start)
                break

    # 按数据偏移排序并去重
    uniq: dict[str, FileEntry] = {}
    for e in sorted(entries, key=lambda x: x.data_offset):
        uniq.setdefault(e.name, e)
    return list(uniq.values())


def extract_stored(buf: bytes, entry: FileEntry) -> bytes | None:
    """取出「仅存储」方式的文件数据。压缩过的返回 None。"""
    if entry.method != 0:
        return None
    end = entry.data_offset + entry.data_size
    if end > len(buf):
        return None  # 数据被片段边界截断
    return buf[entry.data_offset : end]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="扫描 RAR5 片段里的文件条目")
    ap.add_argument("--slice", required=True, help="下载下来的中间片段文件")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    data = open(args.slice, "rb").read()
    found = find_entries(data)
    print(f"片段大小 {len(data) / 1e6:.1f} MB，解析出 {len(found)} 个文件条目\n")
    methods = {}
    for e in found:
        methods[e.method] = methods.get(e.method, 0) + 1
    print(f"压缩方式分布: {methods}  (0 = 仅存储，可直接提取)\n")
    for e in found[: args.limit]:
        complete = e.data_offset + e.data_size <= len(data)
        print(f"  {e.name.split('/')[-1].split(chr(92))[-1]:<20} "
              f"方式={e.method} 压缩后={e.data_size / 1e6:>7.2f}MB "
              f"原始={e.unpacked_size / 1e6:>7.2f}MB "
              f"{'完整' if complete else '被截断'}")
