import math


class Helper:
    # 2018-11-13：手动完成SMA，EMA公式，效率比调用TALIB高。
    # EMA=2*X/(N+1)+(N-1)*EMA(N-1)/(N+1)
    @staticmethod
    def EMA(X, N):
        result = 0
        for power in range(N):
            index = -1*(power+1)
            result += 2.0/(N+1)*pow((N-1)/(N+1), power)*X[index]
        return result

    # SMA(N)=SMA(N-1)*(N-M)/N+X(N)*M/N
    @staticmethod
    def SMA(X, N, M):
        result = 0
        for power in range(len(X)):
            index = -1*(power+1)
            result += (M/N)*pow((N-M)/N, power)*X[index]
        return result

    @staticmethod
    def __window(N, size):
        return 0 if N - size < 0 else N - size

    @staticmethod
    def SUM(X, size):
        N = len(X)
        start = Helper.__window(N, size)
        return sum(X[start:N])

    @staticmethod
    def MA(X, size):
        N = len(X)
        start = Helper.__window(N, size)
        return sum(X[start:N])/(N-start)

    @staticmethod
    def STD(X, size):
        mean = Helper.MA(X, size)
        N = len(X)
        start = Helper.__window(N, size)
        squre = 0
        for j in range(start, N, 1):
            squre += math.pow(X[j]-mean, 2)
        return math.sqrt(squre/size)

    @staticmethod
    def HHV(X, size):
        N = len(X)
        start = Helper.__window(N, size)
        return max(X[start:N])

    @staticmethod
    def LLV(X, size):
        N = len(X)
        start = Helper.__window(N, size)
        return min(X[start:N])

    @staticmethod
    def Region(X):
        return X[-1] >= 0

    @staticmethod
    def IsUp(X):
        return X[-2] <= X[-1]

    @staticmethod
    def UpCross(X, thre):
        return (X[-2] < thre and X[-1] >= thre)

    @staticmethod
    def DownCross(X, thre):
        return (X[-2] >= thre and X[-1] < thre)

    @staticmethod
    def RefN(X, N):
        if len(X) <= N:
            return X[0]
        else:
            return X[-N-1]
