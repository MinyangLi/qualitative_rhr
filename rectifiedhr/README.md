1. checkpoint直接软连接到你的checkpoint保存位置
2. pipelines保存不同模型的pipeline
3. results保存结果
4. utils保存了一些必要工具
5. run_**.py主要的运行代码
6. configs.py保存了超参数

首先需要将flux的代码编程支持对image进行sdedit的形式，即img2img。然后稍微修改下run代码中的变量输入。最后运行进行测试，可以修改configs文件中的参数进行调试结果。

zip -r RectifiedHR.zip RectifiedHR