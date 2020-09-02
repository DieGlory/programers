def solution(m, n, puddles):
    now = []
    dp = [[0 for w in range(101)] for h in range(101)]
    _map = [[0 for w in range(101)] for h in range(101)]  # 01
    state = [[0 for w in range(101)] for h in range(101)]  # 012
    dxdy = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    now.append([1, 1])
    new = []
    M = 1000000007
    for x, y in puddles:
        _map[y][x] = 1

    dp[1][1] = 1
    while now:
        for x, y in now:
            for dx, dy in dxdy:
                nx = x + dx
                ny = y + dy
                if nx >= 1 and ny >= 1 and nx <= m and ny <= n and _map[ny][nx] == 0 and state[ny][nx] != 2:

                    dp[ny][nx] = (dp[ny][nx] + dp[y][x]) % M
                    if state[ny][nx] == 0:
                        state[ny][nx] = 1
                        new.append([nx, ny])
        for x, y in now:
            state[y][x] = 2
        now = new[:]
        new.clear()
    return dp[n][m]
print(solution(4,3,[[2, 2]] ))
