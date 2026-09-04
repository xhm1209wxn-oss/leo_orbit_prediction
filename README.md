# 低轨卫星轨道残差预测

本项目用于根据 TLE 历史数据训练模型，并对低轨卫星未来轨道进行残差修正预测。

## 安装

在项目根目录安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## 训练

准备好 `config.yaml` 和卫星列表后，在项目根目录运行：

```powershell
python run_experiment.py --config config.yaml
```

训练结果会保存在新建的时间戳目录 `outputs.../` 中，其中包括模型文件 `models/best_model.pth`。

## 未来预测

使用一次已完成训练的输出目录进行预测：

```powershell
python scripts/predict_future_corrections.py `
  --run-dir outputs20260903-120000-000000 `
  --satellite 12345 `
  --horizons 12 `
  --output-csv results/future_corrections.csv
```

将 `outputs20260903-120000-000000` 替换为实际训练生成的目录，将 `12345` 替换为目标卫星的 NORAD ID。预测结果保存在指定的 CSV 文件中。

## SGP4 对照

如需生成未修正的 SGP4 传播结果：

```powershell
python scripts/propagate_future_sgp4.py `
  --satellite 12345 `
  --horizons 12 `
  --output-csv results/future_sgp4.csv
```
