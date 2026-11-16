"""下载并解压 Oxford-IIIT Pet 数据集（约 800 MB）

    python download_data.py                    # 下载到 data/oxford-iiit-pet/
    python download_data.py --root D:/datasets # 指定其他目录

已下载过的压缩包会自动跳过；中途断开可重新运行，会从断点继续下载。
"""
import argparse
import os
import tarfile
import urllib.request

URLS = [
    # 牛津 VGG 官方地址（两个文件）
    "https://thor.robots.ox.ac.uk/pets/images.tar.gz",        # 约 755 MB
    "https://thor.robots.ox.ac.uk/pets/annotations.tar.gz",   # 约 18 MB
]


def download(url, dst):
    done = os.path.getsize(dst) if os.path.exists(dst) else 0
    req = urllib.request.Request(url, headers={"Range": "bytes=%d-" % done} if done else {})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:          # 已经下载完整
            return
        raise
    if done and resp.status != 206:  # 服务器没有按断点续传返回，重新下载
        done, mode = 0, "wb"
    else:
        mode = "ab"
    total = done + int(resp.headers.get("Content-Length", 0))
    with open(dst, mode) as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print("\r  %s  %.1f / %.1f MB (%.0f%%)" % (os.path.basename(dst), done / 2**20, total / 2**20,
                                                       100 * done / total), end="", flush=True)
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    args = p.parse_args()
    out = os.path.join(args.root, "oxford-iiit-pet")
    os.makedirs(out, exist_ok=True)
    for url in URLS:
        name = os.path.basename(url)
        sub = name.split(".")[0]                       # images / annotations
        if os.path.isdir(os.path.join(out, sub)):
            print("已存在 %s/，跳过" % sub)
            continue
        tgz = os.path.join(out, name)
        print("下载", url)
        download(url, tgz)
        print("解压", tgz)
        with tarfile.open(tgz) as t:
            t.extractall(out)
        os.remove(tgz)
    n_img = len([f for f in os.listdir(os.path.join(out, "images")) if f.endswith(".jpg")])
    print("完成：%s  （images/ 下 %d 张 jpg）" % (out, n_img))


if __name__ == "__main__":
    main()
