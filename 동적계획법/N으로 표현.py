def solution(N, number):
    num_list = []
    for i in range(1, 9):
        temp_list = {int(str(N) * i)}
        for j in range(1,i//2):
            for l in num_list[j-1]:
                for k in num_list[i-j-1]:
                    temp_list.add(l + k)
                    temp_list.add(l - k)
                    temp_list.add(l * k)
                    if k!=0:
                        temp_list.add(l // k)
        if number in temp_list:
            return i
        num_list.append(temp_list)
    return -1

import time

now = time.time()
print(solution(5, 12))
temp=time.time()
print('%.15f' % (temp-now))
print()
#0.009369611740112305
