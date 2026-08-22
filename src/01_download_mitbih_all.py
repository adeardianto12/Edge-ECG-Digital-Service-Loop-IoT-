import os
import wfdb

# 1. 定义存储全库数据的目标文件夹
data_dir = '../data/mitdb'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print(f"📁 数据存放路径已就绪: {data_dir}")
print("🌐 正在连接 PhysioNet 服务器...")
print("⏳ 开始批量下载 MIT-BIH 完整数据库 (共 48 个记录，约 100MB)，请耐心等待...")

# 2. 调用 wfdb 的官方 API，一键拉取 'mitdb' 全库
# 这个函数会自动跳过已经下载过的文件，支持断点续传
wfdb.dl_database('mitdb', dl_dir=data_dir)

print("=" * 40)
print("🎉 下载完成！MIT-BIH 全库数据已成功保存到本地。")
print("=" * 40)