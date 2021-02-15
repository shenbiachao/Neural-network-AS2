# Neural-network-AS2
如下运行用DANN 训练的脚本：bash train(dann).sh train_path test_path csv_path，其中train_path 为训练图片文件夹(比如../input/tranferdata/ train_data)，test_path 为测试图片文件夹(比如../input/tranfer-data/testdata_raw)，csv_path 为csv 文件(比如../input/tranfer-data/testdata_raw/test.csv)。
如下运行用MMD 训练的脚本：bash train(mmd).sh train_path test_path csv_path。
两个运行脚本执行后会自动训练并保存训练结果，之后对feature进行可视化并画图，最后对测试图片进行预测并生成csv 文件。
如下运行测试代码：bash test.py csv_path test_path param1 param2 param3，其中param1，param2，param3 分别是feature extractor，label predictor，domain classifier 三个网络的参数。

如果有任何问题，请邮箱联系我181840013@smail.nju.edu.cn，谢谢！
