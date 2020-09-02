from itertools import permutations


def solution(numbers):
    answer = 0
    num_list = list(numbers)
    case = []

    for i in range(1,len(numbers)+1):
        case+=list(map(int,map("".join, permutations(num_list,i))))

    case = set(case)

    prime_list = [False, False] + [True] * max(case)
    for i in range(2,int(max(case)**0.5)):
        if prime_list[i] == True:
            for j in range(i+i,max(case)+1,i):
                prime_list[j] = False

    for i in case:
        if prime_list[i]:
            answer+=1

    return answer


print(solution("011"))

# from itertools import permutations
# def solution(n):
#     a = set()
#     for i in range(len(n)):
#         a |= set(map(int, map("".join, permutations(list(n), i + 1))))
#     a -= set(range(0, 2))
#     for i in range(2, int(max(a) ** 0.5) + 1):
#         a -= set(range(i * 2, max(a) + 1, i))
#     return len(a)
