# QField-Dyn 轨迹推理界面

上传PDB拓扑与观测XTC，选择T1–T4档位并填写配体残基名，即可调用模型生成轨迹。界面支持任务状态、轨迹旋转、缩放、播放、运动曲线和预测XTC下载。

## 在线访问

演示入口：[QField-Dyn 在线推理](https://environment-gadgets-estimated-hint.trycloudflare.com/)。访问者通过浏览器上传输入，服务器执行模型推理，结果支持播放和XTC下载。

该入口由服务器提供HTTPS连接，访问者无需配置SSH，也无需保持开发者电脑在线。当前采用临时分享地址，有效期取决于演示服务器和入口进程的运行状态；重新启动入口进程后地址会变化。长期部署可使用固定域名和命名隧道。

## 部署

在Linux CUDA服务器安装仓库依赖并从仓库根目录运行：

```bash
python tools/build_runtime.py --output runtime
python app/server.py --config configs/portal.example.json --port 8787
```

服务绑定`127.0.0.1:8787`。在浏览器所在机器设置端口转发：

```bash
ssh -N -L 8787:127.0.0.1:8787 YOUR_SERVER
```

随后访问`http://127.0.0.1:8787/`。复制示例配置为`configs/portal.local.json`可指定模型、运行目录、GPU与任务保存路径。

外部演示访问可在运行HTTP服务的同一服务器启动Cloudflare Tunnel：

```bash
cloudflared tunnel --no-autoupdate --protocol http2 --url http://127.0.0.1:8787
```

分享命令输出的HTTPS地址。HTTP服务继续监听回环地址，隧道将HTTPS请求转发给该服务。正式长期部署采用固定域名并按使用规模配置访问权限与并发资源。

## 输入与输出

T1/T2/T3的观测及预测帧数分别为10→10、80→20、20→80，帧间隔80 ps。T4为10→490，帧间隔1 ns。

PDB与XTC原子顺序一致，配体CONECT连接完整，观测从0 ps开始。输入由蛋白、配体和支持的单原子离子组成，水在上游准备。每个上传文件最大128 MB。

单队列依次执行任务，结果保存在`jobs/<任务ID>/`，带`job`参数的网址可以重开结果。输入缺少未来真值时，运动曲线描述生成轨迹本身；未来准确性采用有真值案例评价。推理调用与命令行共用`predict_trajectory_adapter.py`。
