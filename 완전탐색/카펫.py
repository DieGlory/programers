# def solution(brown, yellow):
#     answer = []
#     all = brown + yellow
#     for i in range(int(all**0.5),2,-1):
#         if all % i == 0:
#             answer = [int(all/i),i]
#             break
#     return answer
# print(solution(50,22))

def solution(brown, yellow):
    answer = []
    wh = brown + yellow
    # B = (w*2) + (h-2) *2
    # B = 2w + 2h -4
    #(B+4)/2 = w+h
    # Y = (w-2) * (h-2)

    for i in range((brown+4)//2,2,-1):
        if wh % i == 0 and (i-2)*(wh//i-2) == yellow:
            answer = [i,wh//i]
            break
    return answer


# import math
# def solution(brown, yellow):
#     w = ((brown+4)/2 + math.sqrt(((brown+4)/2)**2-4*(brown+yellow)))/2
#     h = ((brown+4)/2 - math.sqrt(((brown+4)/2)**2-4*(brown+yellow)))/2
#     return [w,h]