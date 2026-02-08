# 诊断agent

## 总体输入
1. camel.md
2. task_list.json
3. analysis agent的输出目录
4. task-script的输出目录
5. code agent log 的输出目录
6. ground truth脚本目录

## 每次任务的输入
1. analysis agent得到的一个错误脚本的完整分析报告
2. task描述
3. 错误脚本的完整log内容
4. 生成的task-script文件夹里面这个任务脚本的路径
5. ground truth脚本的路径
（其中给路径是让agent自己决定要不要看，完整内容是一定要给agent看）


## 任务
1. 根据分析报告和code agent执行这个任务的log，分析为什么会出错
2. 修改camel.md,使得下一次code agent载入生成脚本的时候更容易

## agent的实现
1. 在能力层面继承现在的generic_code_agent.py，包括code 能力和分析camel的能力
2. 先阅读log里面explore agent的输出，判断最后的错误是否因为explore agent的输出导致的
3. 如果是，则需要修改camel.md，使得下一次explore agent的输出更容易成功，然后自己载入camel.md重新执行explore agent
4. 如果不是，就跳过explore阶段，直接把log里面explore agent的输出作为code agent的输入，执行code agent，执行自己写camel.md的任务
5. 在任务层面，换成上述任务

## 运行
1. 因为有10个任务，所以你需要运行10次，每次运行后，你需要修改camel.md，使得下一次code agent生成脚本的时候更容易
2. 为了方便调试，也要支持单个任务的调试，即输入一个任务的分析报告、log、task描述、ground truth脚本，输出修改后的camel.md