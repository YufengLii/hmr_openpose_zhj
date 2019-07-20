# hmr_openpose_zhj
结合OpenPose与HMR的实时人体三维建模，附带前端显示
### 安装条件
本文配置方法已在两块RTX2080 以及GTX1080+GTX060的电脑上测试。
- Ubuntu16.04
- 总体显存需要约为10GB+；openpose约3GB

### 配置步骤
1. 显卡驱动安装
2. openpose环境安装：CUDA10.0+cudnn7.3
3. openpose Python API编译
4. HMR环境安装：CUDA9.0+cudnn7.0
5. HMR Python虚拟环境配置
6. ramdisk挂载与文件拷贝
7. openpose+HMR联合环境配置
8. 前端显示配置


### 1 显卡驱动安装
1. 卸载旧驱动
```bash
 sudo apt-get remove --purge nvidia*
 #若安装失败也是这样卸载以及
 ./NVIDIA-Linux-x86_64-390.48.run --uninstall #确保卸载干净。
```
2. 安装可能需要的依赖（可选，脸红可以跳过）
```bash
sudo apt-get update 
 sudo apt-get install dkms build-essential linux-headers-generic
 sudo apt-get install gcc-multilib xorg-dev
 sudo apt-get install freeglut3-dev libx11-dev libxmu-dev install libxi-dev  libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev
```
3. 禁用nouveau 驱动
```sudo vi /etc/modprobe.d/blacklist-nouveau.conf 
#在文件 blacklist-nouveau.conf 中加入如下内容：
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
#保存   :wq
```
```bash
#禁用nouveau 内核模块
echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
sudo update-initramfs -u
reboot #重启
lsmod |grep nouveau #无显示则成功 当然驱动没了你的桌面分辨率会比较大。
```
4. 进入blos关闭secure boot （华硕还有fast boot）
5. 进入tty关闭图形界面
按 CTRL + ALT + F1 键登录，从 GUI 转至终端tty1（全黑色）。为了重建视频输出，必须先将其暂停。
```bash
sudo service lightdm stop
```
6. 运行.run文件选择合适选项
```bash
cd 下载目录
chmod a+x NVIDIA-Linux-x86_64-384.90.run #添加权限
sudo ./NVIDIA-Linux-x86_64-384.90.run --dkms --no-opengl-files
```
- –no-opengl-files：表示只安装驱动文件，不安装OpenGL文件。这个参数不可省略，否则会导致登陆界面死循环，英语一般称为”login loop”或者”stuck in login”。当然脸红的情况下并不会。 
- -dkms（默认开启）在 kernel 自行更新时将驱动程序安装至模块中，从而阻止驱动程序重新安装。在 kernel 更新期间，dkms 触发驱动程序重编译至新的 kernel 模块堆栈

7. 安装过程中的选项
- dkms 安装最好 选yes
- 32位兼容 安装最好 选yes
- x-org 最好别安，选no，有的电脑可能导致登录界面黑屏

8. 安装完成后验证
nvidia-smi #若列出GPU的信息列表，表示驱动安装成功

9. 重新进入桌面
```bash
sudo service lightdm start #没自动跳的话 crtl+alt+f7
nvidia-settings #若弹出设置对话框，亦表示驱动安装成功
```

### 2 openpose环境安装：CUDA10.0+cudnn7.3
openpose 须在CUDA10.0+CUDNN7.3环境下编译；而HMR应使用CUDA9.0+对应的CUDNN版本；CUDA8版本的配置会导致兼容性问题，这里我们先安装CUDA10+cudnn7.3编译openpose,第二个版本的CUDA一定要用runfile的安装方式。
- 首先安装CUDA10+CUDNN7.3。
```bash
sudo chmod +x cuda_10.......................
./cuda_10.0......run
```
**注意：多CDUA安装时都不要创建symbolic link**
```bash
Do you accept the previously read EULA? (accept/decline/quit): accept 
You are attempting to install on an unsupported configuration. Do you wish to continue? ((y)es/(n)o) [ default is no ]: y 
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 346.46? ((y)es/(n)o/(q)uit): n 
Do you want to install the OpenGL libraries? ((y)es/(n)o/(q)uit) [ default is yes ]: n 
Install the CUDA 10.0 Toolkit? ((y)es/(n)o/(q)uit): y 
Enter Toolkit Location [ default is /usr/local/cuda-10.0 ]: 
/usr/local/cuda-10.0 is not writable. 
Do you wish to run the installation with ‘sudo’? ((y)es/(n)o): y 
Please enter your password: 
Do you want to install a symbolic link at /usr/local/cuda? ((y)es/(n)o/(q)uit): n 
Install the CUDA 10.0 Samples? ((y)es/(n)o/(q)uit): y 
Enter CUDA Samples Location [ default is /home/xxx ]: 
Installing the CUDA Toolkit in /usr/local/cuda-10.0 … 
Installing the CUDA Samples in /home/xxx … 
Copying samples to /home/xxx/NVIDIA_CUDA-10.0_Samples now… 
Finished copying samples.
```

cudnnan安装
```bash
# Installing from a Tar File
sudo cp cuda/include/cudnn.h /usr/local/cuda-10.0/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.0/lib64
sudo chmod a+r /usr/local/cuda-10.0/include/cudnn.h /usr/local/cuda-10.0/lib64/libcudnn*
```


环境变量

命令行sudo gedit ~/.bashrc打开.bashrc，末尾加入如下行:

```bash
export PATH="$PATH:/usr/local/cuda-10.0/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64/"
export LIBRARY_PATH="$LIBRARY_PATH:/usr/local/cuda-10.0/lib64" 
```

### 3 openpose Python API 编译
安装过程详情见OPENPOSE GITHUB页面；doc文件夹下有详细描述；有以下几点需要注意
- 使用CMAKE-GUI外部编译；检查configure结果中是否当前CUDA环境为10.0；cudnn环境为7.3；终端输入···nvcc --version···可以查看当前默认的CUDA版本；如果configure版本不对，可在CMAKE-GUII中选在CUDA10.0所在路径
- 选在BUILD Python ；由于HMR在Python2.7下运行，因此openpose也编译为Python2.7的接口
- CMAKE-GUI添加两个变量类型为STRING
``` bash
#变量值根据自己系统里的文件更改，第二个变量值应该稍有不同
PYTHON_EXECUTABLE=/usr/bin/python2.7
PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython2.7m.so
```
- configure 通过即可生成并编译，编译通过后使用sudo make install 将PYTHON API安装到默认路径 ```/usr/local/python/openpose```·之后需要将编译得到的API拷贝到HMR PYTHON虚拟环境的PYTHON路径中，具体位置在步骤7给出

- 编译成功后使用例子进行验证：build/examples/tutorial_api_python/；  openpose_python.py为调用USB摄像头的实时检测代码。可测试安装效果。

- 成功后进入步骤4

### 4 HMR环境安装：CUDA9.0+cudnn7.0
- 安装方法同步骤2
### 5 HMR Python虚拟环境配置
安装方法按照HMR github主页进行，有如下几点需要注意
- 安装HMR虚拟环境前，先将~/.bahsrc下CUDA10.0的环境变量注释；重启或source使其生效
- opendr安装0.77版本；或者从github上自行编译安装，遇到OSMe问题，可参考github issue解决
```bash
pip install opendr==0.77
```
- tensorflow一定不能按照官网安装1.3，应安装1.12版本；
```bash
pip install tensorglow-gpu==1.12
```
- 安装过程中可能会报错，一般是由于本地环境中缺少相应的包导致，如python-tk等，缺什么根据提示google 采用APT或pip方式安装。
- 安装成功后使用官网提供的测试代码进行验证。一张照片约为3分钟甚至更多，耐心等待。速度慢主要是由于渲染时间比较长。
- 成功后进入步骤6

### 6 ramdisk挂载与文件拷贝
### 7 openpose+HMR联合环境配置
### 8 前端显示配置













