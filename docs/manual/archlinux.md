# ArchLinux

## 镜像文件

```
https://mirrors.ustc.edu.cn/archlinux/iso/latest/archlinux-x86_64.iso
```

## 联网
```bash
iwctl # 使用该命令以进行无线连接 进入iwd模式
```

```bash
device list # 列出可用无线网卡设备(可选)
```

```bash
# 扫描并列出可用的 WIFI (可选)
station wlan0 scan 
station wlan0 get-networks
```

```bash
station wlan0 connect xxx # 连接指定 WIFI
```

> 按 Ctrl+C 退出 iwd 模式

```bash
ping www.baidu.com # 测试网络是否可用
```

## 校对时间
```bash
timedatectl set-timezone Asia/Shanghai # 修改时区：
```
### 验证时间：
```bash
date # 输出样例：Sun Dec 25 20:45:32 CST 2022
```

## 分区
```bash
fdisk -l # 列出硬盘和分区情况
```
```bash
cfdisk <硬盘编号> # 进入分区图形化界面 例：cfdisk /dev/sda
```
:::info
个人分区：

/dev/sda:
- /dev/sda1  ->  efi : 1G
- /dev/sda2  ->  swap : 8G
- /dev/sda3  ->  /(根分区) : 247G

/dev/sdb:
- /dev/sdb1  -> /home : 128G
:::
### 格式化分区
```bash
# 系统分区：
mkfs.ext4 <分区编号> # 例：mkfs.ext4 /dev/sda3

# EFI 分区（如果有）：
mkfs.fat -F 32 <分区编号> # 例：mkfs.fat -F 32 /dev/sda1

# 交换分区（如果有）：
mkswap <分区编号> # 例：mkswap /dev/sda2

# 启动交换分区（如果有）：
swapon <分区编号> # 例：swapon /dev/sda2
```
### 挂载分区
```bash
mount <分区编号> /mnt 
```
根据分区情况进行挂载：

```bash
mount /dev/sda3 /mnt # 主分区挂载
mount --mkdir /dev/sdb1 /mnt/home # home分区挂载
mount --mkdir /dev/sda1 /mnt/boot/efi # efi分区挂载
```

## 安装系统

### 更换软件源
```bash
nano /etc/pacman.d/mirrorlist # 也可以使用vim进行编辑
```
**倘若无合适的镜像站则可下载位于中国大陆的HTTPS镜像站：**
```bash
curl -L 'https://archlinux.org/mirrorlist/?country=CN&protocol=https' -o /etc/pacman.d/mirrorlist
```

### 刷新软件包列表
```bash
pacman -Syy
```

### 重新安装`archlinux-keyring`包
```bash
pacman -S archlinux-keyring
```

### 安装基本系统
```bash
pacstrap /mnt base base-devel linux linux-firmware linux-headers

# 建议使用linux-zen内核
pacstrap /mnt base base-devel linux-zen linux-firmware linux-zen-headers
```

### 安装必须软件包
```bash
pacstrap /mnt networkmanager openssh nano vim git grub efibootmgr intel-ucode man-db noto-fonts-cjk 
```
> **可以根据自己的需求进行下载软件包**

## 设置系统
### 生成fstab文件
生成`fstab`文件以使需要的文件系统（如启动目录 /boot）在启动时被自动挂载，用 -U 或 -L 选项分别设置 UUID 或卷标：

```bash
genfstab -U /mnt > /mnt/etc/fstab # 系统启动时会根据生成的 fstab 文件自动挂载分区
```

### chroot 到新安装的系统

接下来的步骤需要像启动到新安装的系统一样直接与其环境、工具和配置进行交互，请 chroot 到新安装的系统：

```bash
arch-chroot /mnt # chroot进入新系统
```
:::tip
此处使用的是`arch-chroot`而不是直接使用`chroot`，注意不要输错了。
:::
### 设置时间和时区
```bash
ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
```
然后运行 `hwclock` 以生成 `/etc/adjtime`：
### 生成 /etc/adjtime：
```bash
hwclock --systohc # 这个命令假定已设置硬件时间为UTC时间
```
### 区域和本地化设置

> **/etc/locale.gen**
```bash
# 去掉前面的 '#'
...
#en_US.UTF-8 UTF-8
#zh_CN.UTF-8 UTF-8
...
```
执行 locale-gen 以生成 locale 信息：
```bash
locale-gen
```
然后创建 locale.conf文件，并编辑设定 LANG 变量

> **/etc/locale.conf**
```bash
LANG=en_US.UTF-8 # LANG 变量如果设置为中文会导致控制台乱码，安装中文字体后图形界面不乱码。
```

### 编辑主机名
```bash
nano /etc/hostname # Cesar
```

### 设置密码
```bash
passwd root # lucian
```

### 安装 grub 引导系统
:::code-group

```bash [传统引导]
# 传统引导：
grub-install --target=i386-pc <硬盘号>
grub-mkconfig -o /boot/grub/grub.cfg
```

```bash [UEFI引导]
# UEFI引导：
grub-install --target=x86_64-efi --efi-directory=/boot/efi --bootloader-id=GRUB
grub-mkconfig -o /boot/grub/grub.cfg
```
:::

### 服务自启动
```bash
systemctl enable sshd
systemctl enable NetworkManager
... # 根据需要自行开启
```
### 创建普通用户
```bash
useradd -m -G wheel <用户名>
```
#### 编辑 sudoers 文件赋予用户管理员权限：
```bash
nano /etc/sudoers
```
> **/etc/sudoers**

```bash
Uncomment to allow members of group wheel to execute any command
%wheel ALL=(ALL:ALL) ALL

Same thing without a password
# 如果想无密码使用sudo 将NOPASSWD:ALL也取消注释
# %wheel ALL=(ALL:ALL) NOPASSWD: ALL
```

### 设置用户密码
```bash
passwd <用户名>
```

## 安装KDE Plasma桌面环境
```bash
sudo pacman -S plasma sddm
```
### sddm开机自启
```bash
sudo systemctl enable sddm
```
### KDE社区提供的应用程序
```bash
sudo pacman -S kde-applications
```
> 该包组包含所有的kde应用程序，不安装kde-applications也要安装kde-utilities包组

## 附加：更新系统（软件包）
```bash
sudo pacman -Syyu
```