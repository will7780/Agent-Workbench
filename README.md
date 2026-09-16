# Agent Workbench

基于 LangGraph ReAct 的本机 Agent 工作台：工具执行、参数校验、人工确认、产物版本绑定和诊断留痕。

**一套通用运行引擎，宿主显式接入自己的工具和规则。**

[English](README.en.md) · [插件边界](docs/PLUGINS.md) · [安全说明](SECURITY.md)

> `0.1.0a1` · MIT · 本机 Alpha。离线演示不调用真实模型，不包含公司业务工具。本项目尚未发布到 PyPI。

## 本地体验

需要 Python 3.10 或更高版本。先安装到独立虚拟环境：

```bash
git clone https://github.com/will7780/Agent-Workbench.git
cd Agent-Workbench
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv/Scripts/python.exe -m pip install .
.venv/Scripts/agent-workbench.exe demo
```

macOS / Linux：

```bash
.venv/bin/python -m pip install .
.venv/bin/agent-workbench demo
```

打开 [对话页](http://127.0.0.1:8786) 和 [诊断页](http://127.0.0.1:8785)。两页由同一个进程和运行实例提供；端口被占用时，可传入 `--chat-port`、`--diagnostics-port`。

在对话页输入“读取演示素材，生成报告并在归档前让我审核”。示例会创建真实的临时报告文件，在审核处暂停。可以批准、拒绝，或要求修改后重新审核；诊断页可查看对应事件、版本、输入和工具反馈。

这是明确标注的离线参考执行器，不是模型能力或真实业务成功率演示。所有示例文件都写入独立演示目录。可用 `--data-dir` 指定目录；默认目录是当前用户的 `~/.agent-workbench/demo/`。

## 保留的机制

- LangGraph 工具调用循环、上下文整理与真实输入角色快照。
- 参数 Schema、可注入的意图校验、权限和人工确认。
- 产物检查、审核、修改后重审和执行版本绑定。
- 结构化 Observation、事件、耗时和 Token；未知资源统计不冒充零。
- 可搜索的工具目录：查看宿主注册的工具、参数合同、风险和声明的运行模式。
- 可钻取的运行流程图：查看实际节点、等待与阻断状态，回查对应事件证据。
- 对话提交后的即时运行提示，在完成、暂停或出错时结束。
- 显式开启的记忆治理、Profile 隔离、Skill 与知识连接器。
- 可选通用 Trace 导出，不依赖测评平台，不自动上传。

## 接入自己的工具

对话页和诊断页顶部均提供“工具目录”和“运行流程”入口。目录内容由宿主插件提供，公开仓库不包含公司私有工具；运行流程依据实际记录生成，节点结束不等于业务验收通过。

工具目录描述输入 Schema、风险和执行模式；插件负责实际业务实现。通过 `RuntimeServices` 显式提供执行器、模型、配置和产物校验器，再用 `AgentRuntime.start/resume/cancel` 执行。

安装你的插件包后，可用 `agent-workbench serve --plugin my_plugin:create_runtime` 启动相同的工作台。真实模型需显式开启，凭据使用 `AGENT_API_ENV_FILE` 指向的中央配置或当前进程变量。网页不安装插件、不上传代码，也不新增密钥管理系统。

完整接口、匿名示例及扩展限制见 [插件接入说明](docs/PLUGINS.md)。支持 JSON/CSV 产物，Excel 解析为可选依赖：`python -m pip install '.[excel]'`。

## 与 Eval 的关系

[E-commerce Eval](https://github.com/will7780/Ecommerce-eval) 是独立的业务验收平台；本项目是 Agent 运行引擎，两者不互相捆绑。`agent-workbench export RUN_ID --output trace.json` 可导出兼容合同 1.2 的诊断 Trace，再由用户选择是否导入测评平台。

运行成功不等于业务正确。工具反馈、产物版本和权限检查是诊断证据，并不自动证明没有发生外部副作用。

## 不包含什么

不包含私有 GUI、业务实现、公司资料、店铺连接器、商品模板或公司规则。也不自研底层图引擎：执行图由 LangGraph 提供。

Python 插件在宿主进程内执行，不是安全沙箱。首版仅用于本机，不承诺远程多用户隔离或跨进程恢复待确认任务。Agent 的执行留痕也不是第三方可信的业务验收证明。

进程重启后保留历史报告，但旧的待确认任务不能恢复；取消是执行节点之间的协作式停止，不会撤销已发生的操作。默认测试只使用离线数据和临时文件，实际模型测试不属于默认验收。

开发与离线验收见 [CONTRIBUTING](CONTRIBUTING.md)，安全边界见 [SECURITY](SECURITY.md)。欢迎提交可复现的匿名问题和插件合同改进。
