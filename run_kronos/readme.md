### 运行模型生成数据
根据 task 启动预测任务，里面有所有信息
用一个类封装，类有函数接受task，然后把结果保存到task的属性里。还有一个参数，决定生图功能，默认开启。生的图方涛result目录。

task可以获取：
1. 哪个模型
2. 输入数据，和未来数据
3. 模型的p和t，默认p=0.9，t=0.6
4. 模型输入输出长度



先检查gpu是否可用，设置模型用cpu还是gpu

task新加一个属性，用来保存运行结果

参考 
[polygon_prediction_example.py](../examples/polygon_prediction_example.py)
[prediction_example.py](../examples/prediction_example.py)
这个代码

