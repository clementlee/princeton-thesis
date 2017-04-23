# maintains a moving average, for tracking errors
class MovingAverage:

    def __init__(self, capacity=10):
        self.capacity = capacity
        self.log = [0.0] * capacity
        self.n = 0

    def addval(self, val):
        self.n += 1
        self.log[self.n % self.capacity] = val   

    def getavg(self):
        num = min(self.capacity, self.n)
        if num == 0:
            return 9e9

        total = float(sum(self.log))

        return total/float(num)


