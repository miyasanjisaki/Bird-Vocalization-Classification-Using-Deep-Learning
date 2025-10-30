# 桌面端可执行程序说明

该目录包含基于 PySimpleGUI 的桌面应用脚本 `gui_app.py`，可以在 Windows 下通过 [PyInstaller](https://pyinstaller.org/) 打包为可执行文件。应用功能与原有 Streamlit 网页版保持一致：

- 选择模型权重、标签目录与单个音频文件
- 设置置信度阈值并执行鸟类叫声识别
- 查看统计信息与事件明细
- 将结果导出为 CSV 文件

## 运行环境

1. 安装 Python 3.9+（建议使用与训练/推理环境一致的版本）。
2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

   如果没有全局的依赖列表，可以单独安装：

   ```bash
   pip install PySimpleGUI pandas torch librosa numpy
   ```

## 直接运行

```bash
python gui_app.py
```

## 打包为 Windows 可执行文件

1. 安装 PyInstaller：

   ```bash
   pip install pyinstaller
   ```

2. 在项目根目录执行打包命令：

   ```bash
   pyinstaller --noconsole --onefile --add-data "train;train" desktop_app/gui_app.py
   ```

   - `--noconsole` 隐藏命令行窗口，适合最终用户使用。
   - `--add-data "train;train"` 会把 `train` 模块打包进去，确保模型加载所需的代码可用。
   - 如果模型权重或标签目录放在固定位置，可根据需要追加 `--add-data` 选项。

3. 成功后，可执行文件位于 `dist/gui_app.exe`。

> 提示：首次运行时会加载模型，耗时较长。模型和标签路径会缓存在输入框中，方便再次识别。
