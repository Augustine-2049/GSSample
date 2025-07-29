# rec

20250725
- 原生3dgs-old代码，跑bicycle，输出点数，不分裂
- 压缩3DGS存储空间：
    - point_video.zip_point_cloud(num_to_keep, src_path, tar_path)
    - 1个点248B -> 43B
    - 思路：控制空间4GB，理论可以控制9E7个点，由于我们后面要采样，先通过opacity在这里控制1E7个点
- 读取video相机位姿：
    - point_video.get_video_cam_info(json_path = os.getcwd())
    - 返回一个x轴俯仰角 + 一组相机位姿矩阵
- 软光栅
    - 实现CPU渲染，并合成视频，1E6个点，3.2s/张
    - 实现GPU渲染，1E6个点，25fps

20250726
- 服务器部署3dgs，跑bicycle得到2E6的场景
- 硬光栅
    - 实现GLSL渲染直出视频，1E6个点，245fps
    - 1080p -> 60fps
    - 点分裂
- point_render.py后续测试CPU下的正确性，point_render_pipeline.py用于测试GLSL管线

待完成
- [ ] 服务器部署，得到一个1E7场景


## 部署
- 使用安装好的gsn2环境


### 部署问题
- Conda 和 pip 混用可能导致元数据不一致（尤其是 numpy 这类基础库）。

- 建议在 Conda 环境中优先使用 conda install 而非 pip install

```bash
AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'

conda install "Pillow<10.0.0"

----------------------------------------
TypeError: No loop matching the specified signature and casting was found for ufunc greater

conda install numpy==1.23.5  # 



```
