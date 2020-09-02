def solution(money):
    length =len(money)
    house1 = [0 for i in range(length)]
    house1[0] = money[0]
    house1[1] = house1[0]
    for i in range(2,length-1):
        house1[i] = max(house1[i-1],house1[i-2]+money[i])

    house = [0 for i in range(length)]
    house[1] = money[1]
    for i in range(2,length):
        house[i] = max(house[i-1],house[i-2]+money[i])

    return max(max(house),max(house1))


print(solution([1, 1000, 1,1,1000,1,1000,100000]))