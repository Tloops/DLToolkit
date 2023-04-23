# tqdm用法

为了方便好看地可视化训练进程，`tqdm`（全称为taqadum，是阿拉伯语，意为进展）就可以用于在循环中动态地显示进度条。



## 0 引入`tqdm`包

```python
from tqdm import tqdm
```



## 1 quickstart

```python
from tqdm import tqdm

for i in tqdm(range(100)):
    # do something
```

`tqdm`表现出来就是内嵌了迭代器，每次循环迭代的`i`和原来只有`range(100)`的值相同

为了简便，`tqdm`还提供了`trange(x)`方法来简化`tqdm(range(x))`，比如上面的也可以写为

```python
from tqdm import trange

for i in trange(100):
    # do something
```

执行过程中效果如下

```
100%|██████████| 100/100 [00:10<00:00, 9.95it/s]
```



## 2 手动方法

### 2.1 set_description

设置进度条前的描述文字，主要用于循环中手动调整描述

```python
from tqdm import tqdm

pbar = tqdm(range(100))
for i in pbar:
    pbar.set_description("Processing image %03d" % i)
    # do something (maybe process some image)
```



### 2.2 update

手动推进进度条，不再使用for来直接迭代tqdm对象

```python
from tqdm import tqdm

pbar = tqdm(total=100) # 总长度为100
for i in range(100):
    # do something (maybe process some image)
    pbar.update(1) # 每次更新1
pbar.close() # 手动关闭占用资源
```



### 2.3 set_postfix

设置进度条之后的描述文字，主要是用于添加一下必要信息

```python
from tqdm import tqdm
from random import random, randint

pbar = tqdm(range(100))
for i in pbar:
    # do something (maybe train one iteration)
    pbar.set_postfix(loss=random(), gen=randint(1, 999), str="h", lst=[1,2])
```

执行过程中效果如下

```
100%|██████████| 100/100 [00:10<00:00, 9.95it/s, gen=95, loss=0.214, lst=[1,2], str=h]
```



## 3 深度学习中的应用

```python
from tqdm import tqdm
from torch.utils.data import DataLoader

dataset = MyDataset()
train_dataloader = DataLoader(dataset)
max_epochs = 100
for epoch in range(max_epoch):
    train_bar = tqdm(train_dataloader)
    for step, data in enumerate(train_bar):
        # model, optim, loss, and so on...
    	train_bar.set_description("iteration [%03d/%03d]" % (step, len(train_dataloader))
    	train_bar.set_postfix(loss=loss.item())
```

效果：

```
iteration [112/112] 100%|██████████| 112/112 [00:14<00:00, 7.91it/s, loss=0.944]
```

