import collections
def bfs(n):
    """
    n 为总血量/最小伤害值。直接用宽度优先搜索暴力模拟过程解除出路径，复杂度O(2^n)。n别太大！！
    里面有个mult是暴击后倍率，自己在代码里面改，原来是2
    """
    mult = 2
    yichu = 0
    ganghao = 0
    time_count = collections.defaultdict(int)
    q = collections.deque([(1, n - 1, [1]), (1, n - mult, [mult])])
    while q:
        time, res, path = q.popleft()
        if res <= 0:
            time_count[time] += 1
            if res < 0:
                yichu += 1
                print("伤害溢出了")
            else:
                ganghao += 1
                print("刚好")
            print(path)
        else:
            q.append((time + 1, res - 1, path + [1]))
            q.append((time + 1, res - mult, path + [mult]))
    s = 0
    t_cnt = 0
    for k, v in time_count.items():
        s += k * v
        t_cnt += v
    print("平均次数：" + str(s/t_cnt))
    print("溢出次数" + str(yichu))
    print("伤害刚好打死人" + str(ganghao))


bfs(10)