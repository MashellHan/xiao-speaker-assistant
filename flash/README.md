# OH2P 安全刷机与音箱端接入指南

适用机型：**Xiaomi 智能音箱 Pro (`OH2P`)**（底部自带 Type-C 接口，免拆机）。

## 安全红线

1. **禁止跨版本盲刷**：不要把旧版 `1.58.6` 的补丁镜像直接刷入 `1.62.x` 或更高版本出厂的音箱，否则会导致原厂小爱语音合成与联网异常。
2. **先双读备份，再解包打补丁**：
   - 使用 Apple Silicon Mac + 数据 Type-C 线捕获 Amlogic WorldCup 模式（`0x1b8e:c003`），窗口约 1.6 秒。
   - 对 `boot0` / `boot1` / `system0` / `system1` 各读取两遍并比对 SHA-256。
   - 从当前正在使用的 `system` 分区直接提取原厂 SquashFS rootfs（读取 superblock `bytes_used`），确保 100% 同版本匹配。
3. **最小化补丁原则**：
   - 仅关闭 Dropbear 密码登录并写入专用 `RSA-4096` 公钥；
   - 在 `/etc/rc.local` 挂载 `/data/open-xiaoai` 启动脚本；
   - 不改动原厂小爱核心二进制与分区表。
4. **单次写入 + 读回校验**：
   - 只刷写当前正在使用的 `system` 分区（`boot0->system0` 或 `boot1->system1`）；
   - 按镜像字节长度读回校验 SHA-256 一致后再 reset。

## 准备阶段命令

```bash
# 1. 检查 Homebrew 依赖（squashfs、libusb-compat、uv）
python3 flash/check_mac.py

# 2. 用数据 Type-C 线连接音箱与 Mac（纯只读探测）
python3 flash/usb_probe.py
```

> 提示：macOS 首次连接新 USB 设备会弹出「允许配件连接」，在刷机抓窗口前可先在「系统设置 → 隐私与安全性」中临时允许，避免 1.6 秒窗口内弹窗拦截。
