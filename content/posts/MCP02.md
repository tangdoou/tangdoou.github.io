---
title: "十分钟上手 MCP "
date: 2025-06-12T10:00:00+08:00 
draft: false # 设置为 false 来发布，true 则为草稿，不会显示在最终网站上
authors: ["唐豆"] # 作者
tags: ["分享", "MCP"] # 标签
categories: ["学习"] 
math: true
summary: "无" 

ShowToc: true
TocOpen: false 
---

## 前言

MCP（模型上下文协议，Model Context Protocol）是 2024 年末由 Claude 的母公司 Anthropic 提出的一个协议。简单来说，它能让大模型连接并使用外部的工具和服务，极大地扩展了模型的能力。

本文将介绍两种推荐的 MCP 部署方法，它们都非常简单易上手，可玩性很高。

看了一圈，第一种方法在简中社区和论坛中还没有人分享，我写的还是挺有参考价值的。

## 方法一：Claude 客户端 + Smithery.ai

Claude 作为 MCP 规则的提出者和制定者，其客户端对 MCP 的支持是**原生级别**的，融合度最高，几乎没有 Bug（其他大模型的客户端目前还没有适配特别好），而且Claude界面非常美观，比 GPT 的好太多了  

#### 1. 下载 Claude 官方客户端

首先，前往 Claude 官网下载桌面客户端。

[https://www.claude.ai/apps](https://www.claude.ai/apps)

![Claude 官网下载页面](https://tc.z.wiki/autoupload/EPCrGlomy_dW_TeigVD2VjbX6Z0L9jPJG2fnSMjU_pGyl5f0KlZfm6UsKj-HyTuv/20250612/IxWy/3306X1984/image.png)

#### 2. 登录 Claude 账号

下载安装后，登录 Claude 账号。

> **注意**：目前 Claude 账号需要使用非中国大陆的手机号进行注册。可以用 `sms-activate` 获取印尼的手机号，成本低。

#### 3. 访问 Smithery.ai 并选择 MCP

登录客户端后，打开网站 [https://smithery.ai/](https://smithery.ai/)。

你可以在这里找到许多社区贡献的 MCP 工具，可以先从首页推荐的开始玩起。

![Smithery.ai 首页](https://tc.z.wiki/autoupload/EPCrGlomy_dW_TeigVD2VjbX6Z0L9jPJG2fnSMjU_pGyl5f0KlZfm6UsKj-HyTuv/20250612/69Gf/3282X1970/image.png)

推荐“谷歌学术”的 MCP ，点击进入详情页。

![谷歌学术 MCP 详情页](https://tc.z.wiki/autoupload/EPCrGlomy_dW_TeigVD2VjbX6Z0L9jPJG2fnSMjU_pGyl5f0KlZfm6UsKj-HyTuv/20250612/6BnO/2982X1964/image.png)

#### 4. 获取并运行安装命令

在 MCP 详情页右侧，点击 `Claude Desktop` 按钮，复制弹出的命令。


![3008X1972/image.png](https://tc.z.wiki/autoupload/EPCrGlomy_dW_TeigVD2VjbX6Z0L9jPJG2fnSMjU_pGyl5f0KlZfm6UsKj-HyTuv/20250612/usQd/3008X1972/image.png)

然后打开你电脑的**终端**（Windows 用户是 PowerShell 或 CMD，macOS 用户是 Terminal），粘贴刚刚复制的命令并回车运行。这个过程 Claude 客户端不需要做任何操作，贼方便。

#### 5. 重启客户端并验证

命令运行成功后，**完全退出并重启 Claude 客户端**。然后打开设置（Settings），你就能看到 MCP 服务已经成功加载了。

![设置中看到已加载的 MCP](https://tc.z.wiki/autoupload/EPCrGlomy_dW_TeigVD2VjbX6Z0L9jPJG2fnSMjU_pGyl5f0KlZfm6UsKj-HyTuv/20250612/DydA/1584X1120/image.png)

#### 6. 如何使用

在 Claude 的对话输入框下方，就可以看到并选择已启用的 MCP 工具了。

![在对话框中使用 MCP](https://tc.z.wiki/autoupload/EPCrGlomy_dW_TeigVD2VjbX6Z0L9jPJG2fnSMjU_pGyl5f0KlZfm6UsKj-HyTuv/20250612/sgmP/2000X1600/image.png)

至此，使用 Claude 客户端的流程就全部完成了。整个过程非常简单，唯一的门槛在于 Claude 账号的注册。

## 方法二：Cherry Studio + 魔搭（ModelScope）

第二种方法对国内用户更友好，使用的是 `Cherry Studio` 客户端和阿里的 `魔搭` 社区，这样就可以用任何的 LLM 使用 MCP 了。缺点是像硬插上去的，有点不是那么原生的融合。

首先，请自行前往官网下载并安装 [Cherry Studio](https://www.cherry-ai.com/)。

然后，打开魔搭社区的 MCP 广场：[https://www.modelscope.cn/mcp](https://www.modelscope.cn/mcp)

![魔搭 MCP 广场](https://tc.z.wiki/autoupload/EPCrGlomy_dW_TeigVD2VjbX6Z0L9jPJG2fnSMjU_pGyl5f0KlZfm6UsKj-HyTuv/20250612/MqJp/3456X2158/Microsoft_Edge_2025-06-12_12.38.42.png)

登录后，找一个你感兴趣的 MCP 点进去，比如“高德地图”

![高德地图 MCP](https://tc.z.wiki/autoupload/EPCrGlomy_dW_TeigVD2VjbX6Z0L9jPJG2fnSMjU_pGyl5f0KlZfm6UsKj-HyTuv/20250612/OPWw/3250X1906/image.png)

关于具体的配置步骤，魔搭社区和 Cherry Studio 官方已经提供了非常详尽的教程。为了避免重复造轮子，这里直接推荐大家参考官方文档，写得比我好（写一半才想起来）。

*   **官方教程链接**：[添加 ModelScope MCP 服务器 | CherryStudio](https://docs.cherry-ai.com/advanced-basic/mcp/tian-jia-modelscope-mcp-fu-wu-qi)

这里放一张使用的案例吧，是用高德的导航，从暨大本部到番禺的驾车路线

![3456X2158/image.png](https://tc.z.wiki/autoupload/EPCrGlomy_dW_TeigVD2VjbX6Z0L9jPJG2fnSMjU_pGyl5f0KlZfm6UsKj-HyTuv/20250612/sG29/3456X2158/image.png)

另外使用 Cherry Studio 调用 MCP 挺慢的，要比 Claude 慢很多。

## 总结

以上就是两种快速上手 MCP 的方法：
1.  **Claude + Smithery.ai**：原生支持，体验稳定流畅，界面党首选，但有账号门槛。
2.  **Cherry Studio + 魔搭**：对国内用户友好，MCP 资源丰富接地气，配置过程有完善的中文文档支持。



最后说一句，个人对于目前的 MCP 看法是一般般的（只是说目前），应用的场景不多，而且体验一般般。但是从魔搭的 MCP 广场可以看出来，国内的很多大企业（甚至有中国银联和支付宝）都有放 MCP 在上边，趋势还是好的。