def solution(phone_book):
    phone_book.sort(key = lambda x : len(x))
    for i in range(len(phone_book)):
        for j in range(i+1,len(phone_book)):
            if phone_book[j].startswith(phone_book[i]):
                return False
    return True
print(solution(['123', '456', '789']))