# 多种采样方案对比

## python环境
```
matplotlib
numpy
tqdm
rdkit
```

## 测试方法
```
python exp_multi_process.py \
  --graph_dir ./data/P-L-graphs \
  --raw_data_dir ./data/P-L \
  --castp_zip_dir ./data/CASTp \
  --sampling mcmc \
  --split_file Splits/test_pocket.txt \
  --split test

P-L为官方数据集
P-L-graphs为第一阶段处理完的图结构数据
CASTp为CASTp提供的标准口袋
```
## 绘制图案
```
python plot_metric_cdf.py \
  --metric precision \
  --scheme MCMC=./results/mcmc_test_results.json \
  --scheme greedy=./results/greedy_test_results.json \
  --scheme graph_cut=./results/graph_cut_test_results.json  \
  --scheme spectral=./results/spectral_test_results.json \
  --output ./pic/precision_cdf.png
```