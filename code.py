# 对每个文件单独使用一个 Vim 会话来编辑，很容易出现冲突的情况，所以你迟早会遇到“已经存在交换文件！”（Swap file “…” already exists!）的错误提示。出现这个提示，有两种可能的原因：
#
# 1、你上次编辑这个文件时，发生了意外崩溃。
#
# 2、你已经在使用另外一个 Vim 会话编辑这个文件了。
#
# 原因不同，我们处理的策略自然也不相同。当进程 ID（process ID）后面没有“STILL RUNNING”这样的字样时，那就是情况 1；否则，就是情况 2 了。
