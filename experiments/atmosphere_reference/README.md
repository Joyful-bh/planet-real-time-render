# 大气参考原型

此目录保存重定位前的球壳 Rayleigh/Mie 单次散射实现，仅用于数学对照、性能基准和后续 LUT 验证。它使用局部平面地面和旧相机模型，不是行星渲染器的稳定入口，禁止在此增加地形、海洋或云功能。

```bash
python -m experiments.atmosphere_reference.cli --preset day --backend auto --preview
```
