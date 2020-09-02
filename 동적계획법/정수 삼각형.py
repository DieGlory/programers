def solution(triangle):
    for i in range(len(triangle)):
        if i == 0:
            continue
        length = len(triangle[i])
        for j in range(length):
            if j == 0:
                triangle[i][j] += triangle[i-1][j]
            elif j == length-1:
                triangle[i][j] += triangle[i-1][j-1]
            else:
                if triangle[i-1][j] > triangle[i-1][j-1]:
                    triangle[i][j] += triangle[i-1][j]
                else:
                    triangle[i][j] += triangle[i - 1][j-1]
    return max(triangle[-1])

print(solution([[7], [3, 8], [8, 1, 0], [2, 7, 4, 4], [4, 5, 2, 6, 5]]))