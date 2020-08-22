import itertools

def solution(numbers):
    answer = ''
    numbers.sort(reverse=True)
    numbers.sort(key=lambda x: len(str(x)))
    return temp[-1]

print(solution([3, 30, 34, 5, 9]))