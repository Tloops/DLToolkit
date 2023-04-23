# Conda常用指令

`conda` 是`python`的常用环境及包管理工具

`conda` is an environment and package management tool for `python`



## 1 查看当前conda版本

```bash
conda --version
```



## 2 创建名为\<name\>、py版本为x.x的环境

```bash
conda create -n <name> python=x.x
```

`<name>`记得替换成你需要的名字，下同

x.x请输入具体的python版本号，如3.7等



## 3 查看当前已创建的环境列表

```bash
conda env list
```



## 4 激活名为\<name\>的环境

```bash
conda activate <name>
```

激活后，可以用下面的命令查看当前环境的python版本：

```bash
python --version
```



## 5 激活后的相关操作

1. 在当前环境**安装**名为`<package_name>`的包：

```bash
conda install <package_name>
```

​	这里的`<package_name>`应替换为具体的包名，如`numpy`等，下同

2. **列出**当前所有已安装的包

```bash
conda list
```

3. **更新**名为`<package_name>`的包

```bash
conda update <package_name>
```

4. **删除**名为`<package_name>`的包

```bash
conda remove <package_name>
```

5. 将当前环境配置**导入yaml文件**

```bash
conda env export > environment.yaml
```



## 6 退出当前环境

```bash
conda deactivate
```



## 7 删除名为\<name\>的环境以及其中的所有包

```bash
conda remove -n <name> --all
```



## 8 根据已有yaml文件创建环境

```bash
conda env create -f environment.yaml
```

