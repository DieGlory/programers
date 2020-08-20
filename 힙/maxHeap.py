

class MaxHeap:
    def __init__(self):
        self.queue = []

    def insert(self,n):
        self.queue.append(n)
        last_index = len(self.queue) - 1

        while 0<= last_index:
            parent_index = self.parent(last_index)
            if 0 <= parent_index and self.queue[parent_index] < self.queue[last_index]:
                self.swap(last_index,parent_index)
            else:
                break

    def delete(self):
        last_index = len(self.queue) -1
        if last_index <0:
            return -1
        self.swap(0,last_index)
        maxv = self.queue.pop()
        self.maxHeapify(0)
        print(maxv)
        return maxv

    def maxHeapift(self,i):
        left_index = self.leftchild(i)
        right_index = self.rightchild(i)
        max_index = i

        if left_index <= len(self.queue) -1 and self.queue[max_index] < self.queue[left_index]:
            max_index = left_index
        if right_index <= len(self.queue) -1 and self.queue[max_index] < self.queue[right_index]:
            max_index = right_index


    def swap(self,i,parent_index):
        self.queue[i], self.queue[parent_index] = self.queue[parent_index],self.queue[i]


    def parent(self,index):
        return (index-1) // 2

    def leftchild(self,index):
        return index*2 +1

    def rightchild(self,index):
        return index*2 +2

    def printHeap(self):
        print(self.queue)

if __name__ == "__main__":
    mh = MaxHeap()
    mh.insert(1)
    mh.insert(3)
    mh.insert(11)
    mh.insert(6)
    mh.insert(5)
    mh.insert(2)
    mh.printHeap()
    mh.delete()
    mh.printHeap()
    mh.delete()
    mh.printHeap()
    mh.delete()
