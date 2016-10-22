# alphaFM_softmax
## 前言：
* alphaFM_softmax是[alphaFM](https://github.com/CastellanZhang/alphaFM)的多分类版本。<br>

* 当将dim参数设置为1,1,0时，alphaFM_softmax就退化成标准的softmax的FTRL训练工具。<br>

##安装方法：
直接在根目录make即可，编译后会在bin目录下生成两个可执行文件。如果编译失败，请升级gcc版本。
##输入文件格式：
类似于libsvm格式，但更加灵活：特征编号不局限于整数也可以是字符串；特征值可以是整数或浮点数（特征值最好做归一化处理，否则可能会导致结果为nan），
特征值为0的项可以省略不写；label必须是1到k（假设类别数为k）。举例如下：<br>
`1 sex:1 age:0.3 f1:1 f3:0.9`<br>
`3 sex:0 age:0.7 f2:0.4 f5:0.8 f8:1`<br>
`2 sex:0 age:0.2 f2:0.6 f8:1`<br>
`...`<br>
##模型文件格式：
假定v的维度为f，类别数为k，类似于alphaFM的格式，只是feature_name之后的部分从1段变成了k段<br>
第一行是bias的参数：<br>
`bias w_1 w_n_1 w_z_1 w_2 w_n_2 w_z_2 ... w_k w_n_k w_z_k`<br>
其他行的格式为：<br>
`feature_name w_1 v1_1 v2_1 ... vf_1 w_n_1 w_z_1 v_n1_1 v_n2_1 ... v_nf_1 v_z1_1 v_z2_1 ... v_zf_1 w_2 v1_2 v2_2 ... vf_2 w_n_2 w_z_2 v_n1_2 v_n2_2 ... v_nf_2 v_z1_2 v_z2_2 ... v_zf_2 ... ... w_k v1_k v2_k ... vf_k w_n_k w_z_k v_n1_k v_n2_k ... v_nf_k v_z1_k v_z2_k ... v_zf_k`
##预测结果格式：
`label score_1 score_2 ... score_k`<br>
label为输入数据的类别标注值（1到k），score_i为样本属于类别i的预测概率值。

##参数说明：
###fm_train_softmax的参数：
和alphaFM基本一致，多了一个-cn参数，少了一个-fvs参数<br>
-m \<model_path\>: 设置模型文件的输出路径。<br>
-cn \<class_num\>: 设置类别数。<br>
-dim \<k0,k1,k2\>: k0为1表示使用偏置w0参数，0表示不使用；k1为1表示使用w参数，为0表示不使用；k2为v的维度，可以是0。	default:1,1,8<br>
-init_stdev \<stdev\>: v的初始化使用均值为0的高斯分布，stdev为标准差。	default:0.1<br>
-w_alpha \<w_alpha\>: w0和w的FTRL超参数alpha。	default:0.05<br>
-w_beta \<w_beta\>: w0和w的FTRL超参数beta。	default:1.0<br>
-w_l1 \<w_L1_reg\>: w0和w的L1正则。	default:0.1<br>
-w_l2 \<w_L2_reg\>: w0和w的L2正则。	default:5.0<br>
-v_alpha \<v_alpha\>: v的FTRL超参数alpha。	default:0.05<br>
-v_beta \<v_beta\>: v的FTRL超参数beta。	default:1.0<br>
-v_l1 \<v_L1_reg\>: v的L1正则。	default:0.1<br>
-v_l2 \<v_L2_reg\>: v的L2正则。	default:5.0<br>
-core \<threads_num\>: 计算线程数。	default:1<br>
-im \<initial_model_path\>: 上次模型的路径，用于初始化模型参数。如果是第一次训练则不用设置此参数。<br>
###fm_predict_softmax的参数：
比alphaFM多了一个-cn参数<br>
-m \<model_path\>: 模型文件路径。<br>
-cn \<class_num\>: 设置类别数。<br>
-dim \<factor_num\>: v的维度。	default:8<br>
-core \<threads_num\>: 计算线程数。	default:1<br>
-out \<predict_path\>: 输出文件路径。<br>
##计算速度：
###我的实验结果：
本地1000万的样本，200万的特征维度，7个类别，v的维度为12，2.10GHz的CPU，开10个线程，非缺省参数如下：<br>
`-cn 7 -dim 1,1,12 -w_l1 0.05 -v_l1 0.05 -init_stdev 0.001 -w_alpha 0.01 -v_alpha 0.01 -core 10`<br>
训练时间大概半个多小时。

