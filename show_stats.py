import pstats

#
p = pstats.Stats('p.stats')
p.sort_stats("cumulative")
# 输出累计时间报告
p.print_stats(10)
# 输出调用者的信息
# p.print_callers()
# 输出哪个函数调用了哪个函数
# p.print_callees()
