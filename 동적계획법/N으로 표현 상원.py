def func(y, N):
    y = str(y)
    N = str(N)
    for x in y:
        if x != N: return False
    return True


def solution(N, number):
    answer = 0

    cont = []
    part = []
    check = set()

    cont.append([0, N])

    for idx in range(1, 9):

        for x, y in cont:

            if x == number:
                for i in part:
                    print(*i)
                return idx
            if (x, y) in check: continue
            check.add((x, y))

            cont.append([x + y, 0])
            cont.append([x - y, 0])
            cont.append([x * y, 0])
            if (y != 0): cont.append([x // y, 0])

            part.append([x, y + N])
            part.append([x, y - N])
            part.append([x, y * N])
            part.append([x, y // N])

            if func(y, N): part.append([x, y * 10 + N])

        cont = part[:]

    return -1
import time

now = time.time()
print(solution(5, 12))
temp=time.time()
print('%.15f' % (temp-now))
print()
#0.009369611740112305