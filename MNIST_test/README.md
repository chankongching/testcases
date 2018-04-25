# Command to use to execute job
Mounting the /root/code/Distribute_MNIST path into docker

nvidia-docker run -itd -p 2222:2222 -v /root/code:/root/code -v /root/result:/root/result tensorflow/tensorflow:latest  python /root/code/distributed.py --job_name="ps" --ps_hosts="192.168.1.173:2222" --worker_hosts="192.168.1.171:2222,192.168.1.172:2222" --task_index=0

nvidia-docker run -itd -p 2222:2222 -v /root/code:/root/code -v /root/result:/root/result tensorflow/tensorflow:latest  python /root/code/distributed.py --job_name="worker" --ps_hosts="192.168.1.173:2222" --worker_hosts="192.168.1.171:2222,192.168.1.172:2222"  --task_index=0

nvidia-docker run -itd -p 2222:2222 -v /root/code:/root/code -v /root/result:/root/result tensorflow/tensorflow:latest  python /root/code/distributed.py --job_name="worker" --ps_hosts="192.168.1.173:2222" --worker_hosts="192.168.1.171:2222,192.168.1.172:2222"  --task_index=1

# Expected result
```
NODE 1


1524487685.116428: Worker 0: traing step 8054 dome (global step:9960)
1524487685.160202: Worker 0: traing step 8055 dome (global step:9961)
1524487685.203343: Worker 0: traing step 8056 dome (global step:9962)
1524487685.246421: Worker 0: traing step 8057 dome (global step:9963)
1524487685.292357: Worker 0: traing step 8058 dome (global step:9965)
1524487685.336263: Worker 0: traing step 8059 dome (global step:9966)
1524487685.382817: Worker 0: traing step 8060 dome (global step:9967)
1524487685.424566: Worker 0: traing step 8061 dome (global step:9968)
1524487685.467782: Worker 0: traing step 8062 dome (global step:9969)
1524487685.512781: Worker 0: traing step 8063 dome (global step:9971)
1524487685.554451: Worker 0: traing step 8064 dome (global step:9972)
1524487685.596355: Worker 0: traing step 8065 dome (global step:9973)
1524487685.639382: Worker 0: traing step 8066 dome (global step:9974)
1524487685.684103: Worker 0: traing step 8067 dome (global step:9976)
1524487685.730389: Worker 0: traing step 8068 dome (global step:9977)
1524487685.774431: Worker 0: traing step 8069 dome (global step:9978)
1524487685.818752: Worker 0: traing step 8070 dome (global step:9979)
1524487685.864362: Worker 0: traing step 8071 dome (global step:9981)
1524487685.910818: Worker 0: traing step 8072 dome (global step:9982)
1524487685.955815: Worker 0: traing step 8073 dome (global step:9983)
1524487686.001664: Worker 0: traing step 8074 dome (global step:9984)
1524487686.047667: Worker 0: traing step 8075 dome (global step:9986)
1524487686.091223: Worker 0: traing step 8076 dome (global step:9987)
1524487686.134904: Worker 0: traing step 8077 dome (global step:9988)
1524487686.177393: Worker 0: traing step 8078 dome (global step:9989)
1524487686.221982: Worker 0: traing step 8079 dome (global step:9991)
1524487686.271100: Worker 0: traing step 8080 dome (global step:9992)
1524487686.319529: Worker 0: traing step 8081 dome (global step:9993)
1524487686.365306: Worker 0: traing step 8082 dome (global step:9994)
1524487686.409443: Worker 0: traing step 8083 dome (global step:9996)
1524487686.453835: Worker 0: traing step 8084 dome (global step:9997)
1524487686.497427: Worker 0: traing step 8085 dome (global step:9998)
1524487686.541502: Worker 0: traing step 8086 dome (global step:9999)
1524487686.584212: Worker 0: traing step 8087 dome (global step:10001)
Training ends @ 1524487686.584364
Training elapsed time:366.604071 s
After 10000 training step(s), validation cross entropy = 1147.88
```

```
NODE 2

1524487681.483467: Worker 1: traing step 1887 dome (global step:9864)
1524487681.615457: Worker 1: traing step 1888 dome (global step:9869)
1524487681.764761: Worker 1: traing step 1889 dome (global step:9872)
1524487681.941379: Worker 1: traing step 1890 dome (global step:9876)
1524487682.077306: Worker 1: traing step 1891 dome (global step:9881)
1524487682.226872: Worker 1: traing step 1892 dome (global step:9885)
1524487682.375240: Worker 1: traing step 1893 dome (global step:9889)
1524487682.578290: Worker 1: traing step 1894 dome (global step:9893)
1524487682.779062: Worker 1: traing step 1895 dome (global step:9898)
1524487682.971550: Worker 1: traing step 1896 dome (global step:9902)
1524487683.193175: Worker 1: traing step 1897 dome (global step:9907)
1524487683.435355: Worker 1: traing step 1898 dome (global step:9913)
1524487683.599336: Worker 1: traing step 1899 dome (global step:9919)
1524487683.760519: Worker 1: traing step 1900 dome (global step:9923)
1524487683.929460: Worker 1: traing step 1901 dome (global step:9927)
1524487684.096199: Worker 1: traing step 1902 dome (global step:9932)
1524487684.260782: Worker 1: traing step 1903 dome (global step:9937)
1524487684.441629: Worker 1: traing step 1904 dome (global step:9941)
1524487684.595703: Worker 1: traing step 1905 dome (global step:9946)
1524487684.784011: Worker 1: traing step 1906 dome (global step:9950)
1524487684.976657: Worker 1: traing step 1907 dome (global step:9953)
1524487685.189852: Worker 1: traing step 1908 dome (global step:9958)
1524487685.403831: Worker 1: traing step 1909 dome (global step:9964)
1524487685.606131: Worker 1: traing step 1910 dome (global step:9970)
1524487685.776894: Worker 1: traing step 1911 dome (global step:9976)
1524487685.945085: Worker 1: traing step 1912 dome (global step:9981)
1524487686.128778: Worker 1: traing step 1913 dome (global step:9985)
1524487686.308813: Worker 1: traing step 1914 dome (global step:9990)
1524487686.474578: Worker 1: traing step 1915 dome (global step:9995)
1524487686.657338: Worker 1: traing step 1916 dome (global step:10000)
Training ends @ 1524487686.657494
Training elapsed time:366.428102 s
After 10000 training step(s), validation cross entropy = 1122.46
```

```
PS node
/usr/local/lib/python2.7/dist-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting MNIST_data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
job_name : ps
task_index : 0
2018-04-23 12:37:55.762710: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-04-23 12:37:55.763640: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:215] Initialize GrpcChannelCache for job ps -> {0 -> localhost:2222}
2018-04-23 12:37:55.763658: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:215] Initialize GrpcChannelCache for job worker -> {0 -> 192.168.1.171:2222, 1 -> 192.168.1.172:2222}
2018-04-23 12:37:55.764195: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:324] Started server with target: grpc://localhost:2222
```
# TensorFlow分布式MNIST手写字体识别实例

## 代码运行步骤
ps 节点执行：

```
python distributed.py --job_name=ps --task_index=0
```

worker1 节点执行：

```
python distributed.py --job_name=worker --task_index=0
```

worker2 节点执行：

```
python distributed.py --job_name=worker --task_index=1
```

## 运行结果
#### worker0节点运行结果
![分布式TF运行结果worker1](https://github.com/TracyMcgrady6/Distribute_MNIST/blob/master/屏幕快照%202017-07-27%20下午5.12.07.png)
#### worker1节点运行结果
![分布式TF运行结果worker1](https://github.com/TracyMcgrady6/Distribute_MNIST/blob/master/屏幕快照%202017-07-27%20下午5.13.16.png)

该例子是TF的入门实例手写字体识别MNIST基于分布式的实现，代码都加了中文注释。更加通俗易懂，下面讲解TensorFlow分布式原理。

## TF分布式原理
TF的实现分为了单机实现和分布式实现，在分布式实现中，需要实现的是对client，master，worker process不在同一台机器上时的支持。数据量很大的情况下，单机跑深度学习程序，过于耗时，所以需要TensorFlow分布式并行。

![单机与分布式结构](https://github.com/TracyMcgrady6/Distribute_MNIST/blob/master/4.png)

## Single-Device Execution
构建好图后，使用拓扑算法来决定执行哪一个节点，即对每个节点使用一个计数，值表示所依赖的未完成的节点数目，当一个节点的运算完成时，将依赖该节点的所有节点的计数减一。如果节点的计数为0，将其放入准备队列待执行。
### 单机多GPU训练
先简单介绍下单机的多GPU训练，然后再介绍分布式的多机多GPU训练。
单机的多GPU训练， tensorflow的官方已经给了一个cifar的例子，已经有比较详细的代码和文档介绍， 这里大致说下多GPU的过程，以便方便引入到多机多GPU的介绍。
单机多GPU的训练过程：

1. 假设你的机器上有3个GPU;

2. 在单机单GPU的训练中，数据是一个batch一个batch的训练。 在单机多GPU中，数据一次处理3个batch(假设是3个GPU训练）， 每个GPU处理一个batch的数据计算。

3. 变量，或者说参数，保存在CPU上

4. 刚开始的时候数据由CPU分发给3个GPU， 在GPU上完成了计算，得到每个batch要更新的梯度。

5. 然后在CPU上收集完了3个GPU上的要更新的梯度， 计算一下平均梯度，然后更新参数。

6. 然后继续循环这个过程。

通过这个过程，处理的速度取决于最慢的那个GPU的速度。如果3个GPU的处理速度差不多的话， 处理速度就相当于单机单GPU的速度的3倍减去数据在CPU和GPU之间传输的开销，实际的效率提升看CPU和GPU之间数据的速度和处理数据的大小。

#### 通俗解释
写到这里觉得自己写的还是不同通俗易懂， 下面就打一个更加通俗的比方来解释一下：

老师给小明和小华布置了10000张纸的乘法题并且把所有的乘法的结果加起来， 每张纸上有128道乘法题。 这里一张纸就是一个batch， batch_size就是128. 小明算加法比较快， 小华算乘法比较快，于是小华就负责计算乘法， 小明负责把小华的乘法结果加起来 。 这样小明就是CPU，小华就是GPU.

这样计算的话， 预计小明和小华两个人得要花费一个星期的时间才能完成老师布置的题目。 于是小明就招来2个算乘法也很快的小红和小亮。 于是每次小明就给小华，小红，小亮各分发一张纸，让他们算乘法， 他们三个人算完了之后， 把结果告诉小明， 小明把他们的结果加起来，然后再给他们没人分发一张算乘法的纸，依次循环，知道所有的算完。

这里小明采用的是同步模式，就是每次要等他们三个都算完了之后， 再统一算加法，算完了加法之后，再给他们三个分发纸张。这样速度就取决于他们三个中算乘法算的最慢的那个人， 和分发纸张的速度。
## Multi-Device Execution
当系统到了分布式情况下时，事情就变得复杂了很多，还好前述调度用了现有的框架。那么对于TF来说，剩下的事情就是：

决定运算在哪个设备上运行
管理设备之间的数据传递
### 分布式多机多GPU训练
随着设计的模型越来越复杂，模型参数越来越多，越来越大， 大到什么程度？多到什么程度？ 多参数的个数上百亿个， 训练的数据多到按TB级别来衡量。大家知道每次计算一轮，都要计算梯度，更新参数。 当参数的量级上升到百亿量级甚至更大之后， 参数的更新的性能都是问题。 如果是单机16个GPU， 一个step最多也是处理16个batch， 这对于上TB级别的数据来说，不知道要训练到什么时候。于是就有了分布式的深度学习训练方法，或者说框架。

### 参数服务器
在介绍tensorflow的分布式训练之前，先说下参数服务器的概念。
前面说道， 当你的模型越来越大， 模型的参数越来越多，多到模型参数的更新，一台机器的性能都不够的时候， 很自然的我们就会想到把参数分开放到不同的机器去存储和更新。
因为碰到上面提到的那些问题， 所有参数服务器就被单独拧出来， 于是就有了参数服务器的概念。 参数服务器可以是多台机器组成的集群， 这个就有点类似分布式的存储架构了， 涉及到数据的同步，一致性等等， 一般是key-value的形式，可以理解为一个分布式的key-value内存数据库，然后再加上一些参数更新的操作。 详细的细节可以去google一下， 这里就不详细说了。 反正就是当性能不够的时候，
 几百亿的参数分散到不同的机器上去保存和更新，解决参数存储和更新的性能问题。
借用上面的小明算题的例子，小明觉得自己算加法都算不过来了， 于是就叫了10个小明过来一起帮忙算。
### gRPC (google remote procedure call)
TensorFlow分布式并行基于gRPC通信框架，其中包括一个master创建Session，还有多个worker负责执行计算图中的任务。

gRPC首先是一个RPC，即远程过程调用,通俗的解释是：假设你在本机上执行一段代码`num=add(a,b)`，它调用了一个过程 call，然后返回了一个值num，你感觉这段代码只是在本机上执行的, 但实际情况是,本机上的add方法是将参数打包发送给服务器,然后服务器运行服务器端的add方法,返回的结果再将数据打包返回给客户端.
### 结构
Cluster是Job的集合，Job是Task的集合。
>即：一个Cluster可以切分多个Job，一个Job指一类特定的任务，每个Job包含多个Task，比如parameter server(ps)、worker，在大多数情况下,一个机器上只运行一个Task.

在分布式深度学习框架中,我们一般把Job划分为Parameter Server和Worker:

* Parameter Job是管理参数的存储和更新工作.
* Worker Job是来运行ops.

如果参数的数量太大,一台机器处理不了,这就要需要多个Tasks.

### TF分布式模式
#### In-graph 模式
>将模型的计算图的不同部分放在不同的机器上执行

In-graph模式和单机多GPU模型有点类似。 还是一个小明算加法， 但是算乘法的就可以不止是他们一个教室的小华，小红，小亮了。 可以是其他教师的小张，小李。。。。

In-graph模式， 把计算已经从单机多GPU，已经扩展到了多机多GPU了， 不过数据分发还是在一个节点。 这样的好处是配置简单， 其他多机多GPU的计算节点，只要起个join操作， 暴露一个网络接口，等在那里接受任务就好了。 这些计算节点暴露出来的网络接口，使用起来就跟本机的一个GPU的使用一样， 只要在操作的时候指定tf.device("/job:worker/task:n")，
 就可以向指定GPU一样，把操作指定到一个计算节点上计算，使用起来和多GPU的类似。 但是这样的坏处是训练数据的分发依然在一个节点上， 要把训练数据分发到不同的机器上， 严重影响并发训练速度。在大数据训练的情况下， 不推荐使用这种模式。
#### Between-graph 模式
>数据并行，每台机器使用完全相同的计算图


Between-graph模式下，训练的参数保存在参数服务器， 数据不用分发， 数据分片的保存在各个计算节点， 各个计算节点自己算自己的， 算完了之后， 把要更新的参数告诉参数服务器，参数服务器更新参数。这种模式的优点是不用训练数据的分发了， 尤其是在数据量在TB级的时候， 节省了大量的时间，所以大数据深度学习还是推荐使用Between-graph模式。

### 同步更新和异步更新

>in-graph模式和between-graph模式都支持同步和异步更新。

在同步更新的时候， 每次梯度更新，要等所有分发出去的数据计算完成后，返回回来结果之后，把梯度累加算了均值之后，再更新参数。 这样的好处是loss的下降比较稳定， 但是这个的坏处也很明显， 处理的速度取决于最慢的那个分片计算的时间。

在异步更新的时候， 所有的计算节点，各自算自己的， 更新参数也是自己更新自己计算的结果， 这样的优点就是计算速度快， 计算资源能得到充分利用，但是缺点是loss的下降不稳定， 抖动大。

在数据量小的情况下， 各个节点的计算能力比较均衡的情况下， 推荐使用同步模式；数据量很大，各个机器的计算性能掺差不齐的情况下，推荐使用异步的方式。

## 实例
tensorflow官方有个分布式tensorflow的文档，但是例子没有完整的代码， 这里写了一个最简单的可以跑起来的例子，供大家参考，这里也傻瓜式给大家解释一下代码，以便更加通俗的理解。
