'''
    顾名思义，贪心算法或贪心思想采用贪心的策略，保证每次操作都是局部最优的，从而使最后得到的结果是全局最优的。
    在贪心算法中，对数组或字符串排序（根据实际选择升序或者降序）是常见的操作，方便之后的大小比较。
    对于多维数组，需要根据实际情况判断按维度的什么先后顺序来进行排序（如435），可以通过在sort()中加入sort(key=lambda x: x[1])来实现。
'''
class Solution(object):
    '''
    455. Assign Cookies(easy)
    假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。
    对每个孩子 i，都有一个胃口值 g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，都有一个尺寸 s[j] 。
    如果 s[j] >= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。

    贪心思想：先对孩子胃口和饼干尺寸分别排序，尽量考虑用小饼干满足胃口小的孩子
    '''
    def findContentChildren(self, g, s):
        """
        :type g: List[int]
        :type s: List[int]
        :rtype: int
        """
        #为小孩和饼干数组都进行排序方便比较，并初始化两个指针
        g.sort()
        s.sort()
        num_kid = 0
        num_cookie = 0
        #开始遍历，如果当前饼干能满足当前小孩，则饼干和小孩的指针都往后移动一位，否则只移动饼干（即换更大的饼干尝试）
        while num_kid < len(g) and num_cookie < len(s):
            if g[num_kid] <= s[num_cookie] :
                num_kid +=1
                num_cookie +=1
            else :
                num_cookie += 1
        return num_kid

    '''
    135. Candy(hard)
    老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。
    你需要按照以下要求，帮助老师给这些孩子分发糖果：
    每个孩子至少分配到 1 个糖果。
    评分更高的孩子必须比他两侧的邻位孩子获得更多的糖果。
    那么这样下来，老师至少需要准备多少颗糖果呢？

    贪心思想：得分高的孩子必须比两侧邻位的糖果多，那就只让他多一颗
    '''
    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """
        #先初始化一个长为小孩数，值都为1的列表，即每个小朋友都发一颗糖
        candy = list(1 for each in range(len(ratings)))
        #从左往右遍历，如果右边的小孩得分更高但是糖更少，把其糖果数置为左边小孩+1
        for i in range(len(ratings)-1) :
            if ratings[i]<ratings[i+1] and candy[i]>= candy[i+1] :
                candy[i+1] =candy[i] +1
        #同理从右往左再遍历一遍即可，这样即满足了两侧的要求
        for i in range(len(ratings)-1) :
            if ratings[-i-1]<ratings[-i-2] and candy[-i-1]>= candy[-i-2] :
                candy[-i-2] =candy[-i-1] +1
        return sum(candy)

    '''
    435. Non-overlapping Intervals(Medium)
    给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。
    注意:
        可以认为区间的终点总是大于它的起点。
        区间 [1,2] 和 [2,3] 的边界相互“接触”，但没有相互重叠。

    贪心思想：先留下右边界值最小的区间，再去除重复区间
    '''
    def eraseOverlapIntervals(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        if len(intervals) <= 1:
            return 0
        #按右边界的大小升序排列，记录第一个区间为当前区间
        intervals.sort(key=lambda x: x[1])
        pair = intervals[0]
        count = 0
        #遍历每个元素，讲其左边界与当前区间的右边界相比较，若较小则其为需要移除的，若较大则用其来更新当前区间
        for i in range(len(intervals)-1) :
            if  intervals[i+1][0]<pair[1]:
                count +=1
            else :
                pair = intervals[i+1]
        return count

    '''
    605. Can Place Flowers (easy)
    假设有一个很长的花坛，一部分地块种植了花，另一部分却没有。可是，花不能种植在相邻的地块上，它们会争夺水源，两者都会死去。
    给你一个整数数组  flowerbed 表示花坛，由若干 0 和 1 组成，其中 0 表示没种植花，1 表示种植了花。另有一个数 n ，
    能否在不打破种植规则的情况下种入 n 朵花？能则返回 true ，不能则返回 false。

    贪心思想：找到空格数与最大可种花数的关系后遍历所有位置，注意左右边界。
    '''
    def canPlaceFlowers(self, flowerbed, n):
        """
        :type flowerbed: List[int]
        :type n: int
        :rtype: bool
        """
        #初始化可种花数和上一朵花的位置，置为-1是解决左边界问题
        count = 0
        pre_pos = -1
        #遍历每个位置，如果该位置上有花，则通过判断左边界来确定可种花数，简单可以推知：非左边界上三个空一个花，五个空两个，满足 花=(空-1)/2
        #而左边界上，二空一个，四空两个，满足 花=空/2  ，又 空=当前位置-上一朵花位置-1
        for i in range(len(flowerbed)):
            if flowerbed[i]:
                if pre_pos == -1:
                    count += i // 2
                else:
                    count += (i - pre_pos - 2) // 2
                pre_pos = i
        #最后判断是否全空 ，并解决右边界问题
        if pre_pos == -1:
            count += (len(flowerbed) + 1) // 2
        else:
            count += (len(flowerbed) - pre_pos - 1) // 2
        return (count >= n)

    '''
    452. Minimum Number of Arrows to Burst Balloons(Medium)
    在二维空间中有许多球形的气球。对于每个气球，提供的输入是水平方向上，气球直径的开始和结束坐标。由于它是水平的，所以纵坐标并不重要，因此只要知道开始和结束的横坐标就足够了。开始坐标总是小于结束坐标。
    一支弓箭可以沿着 x 轴从不同点完全垂直地射出。在坐标 x 处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 
    且满足  xstart ≤ x ≤ xend，则该气球会被引爆。可以射出的弓箭的数量没有限制。 弓箭一旦被射出之后，可以无限地前进。
    我们想找到使得所有气球全部被引爆，所需的弓箭的最小数量。
    给你一个数组 points ，其中 points [i] = [xstart,xend] ，返回引爆所有气球所必须射出的最小弓箭数。

    贪心思想：十分类似于435题，只是435题是去除重复空间，而这道题可以理解成保留最多不重复空间，所以唯一区别是count计数方式不同，且区间边界定义不同
    '''
    def findMinArrowShots(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        if len(points) <= 1:
            return len(points)
        points.sort(key=lambda x: (x[1], x[0]))
        pair = points[0]
        count = 1
        for i in range(1, len(points)):
            if points[i][0] > pair[1]:
                count += 1
                pair = points[i]
        return count

    '''
    763. Partition Labels (Medium)
    字符串 S 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。返回一个表示每个字符串片段的长度的列表。
    
    贪心思想：因为同一字母最多出现在一个片段中，所以我们在遍历到的每一个字母，都在字符串中反向查找其位置（即找最后出现的位置），如果这个位置超过了当前定义的右区间，
    则用其来更新右区间，若这个位置就是当前右区间，说明我们找到了一个符合条件的区间，将其位置信息保存。
    '''
    def partitionLabels(self, S):
        """
        :type S: str
        :rtype: List[int]
        """
        #初始化，index_list列表中保存的可以认为是区间边界的位置，默认的左区间自然是0，end_index是当前最大的右区间位置
        index_list = [0]
        res = []
        end_index = 0
        #遍历每一个字母，都在字符串中反向查找其位置（即找最后出现的位置），如果这个位置超过了当前定义的右区间，则用其来更新右区间，
        # 若这个位置就是当前右区间或者右边界，说明我们找到了一个符合条件的区间，将其位置信息保存。
        for i in range(len(S)):
            index = S.rindex(S[i])
            if index == i and i == end_index:
                index_list.append(end_index)
                end_index = i + 1
            elif index > end_index:
                end_index = index
        # 计算每一个区间的长度
        for i in range(1, len(index_list)):
            if i == 1:
                res.append(index_list[i] - index_list[i - 1] + 1)
            else:
                res.append(index_list[i] - index_list[i - 1])
        return res
    '''
    122. Best Time to Buy and Sell Stock II (Easy)
    给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
    设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
    注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
    
    贪心思想：既然不限制交易次数，也没有手续费，那我在价格为[3，2，1]时，第一天买第三天卖，和第一天买第二天卖加第二天买第三天卖一样，那就可以简化成只要赚就买，只要赔就不动
    '''
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices) == 1:
            return 0
        profit = []
        #遍历每个位置，只要与前一天相比挣了就把钱保存下来，没挣就不保存
        for i in range(1, len(prices)):
            if prices[i] - prices[i - 1] > 0:
                profit.append(prices[i] - prices[i - 1])
        return sum(profit)

    '''
    406. Queue Reconstruction by Height (Medium)
    假设有打乱顺序的一群人站成一个队列，数组 people 表示队列中一些人的属性（不一定按顺序）。
    每个 people[i] = [hi, ki] 表示第 i 个人的身高为 hi ，前面 正好 有 ki 个身高大于或等于 hi 的人。
    请你重新构造并返回输入数组 people 所表示的队列。返回的队列应该格式化为数组 queue ，其中 queue[j] = [hj, kj] 是队列中第 j 个人的属性（queue[0] 是排在队列前面的人）。
    
    贪心思想：对于高个子来说，前面插入矮个子并不会影响自己的第二个属性ki，故我们可以先把高个子往前排，然后插入矮个子即可
    '''
    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        # 按身高逆序，ki顺序排列，初始化一个空队列
        people.sort(key=lambda x: (-x[0], x[1]))
        queue = []
        # 遍历每个元素，如果当前ki小于队列长，说明前面的高个子数（注意：因为是身高降序遍历，所以前面的都是比自己高或相等的）超过了自身ki，
        # 所以把自己插入相应位置，否则直接加到末尾即可
        for each in people:
            if each[1] >= len(queue):
                queue.append(each)
            elif each[1] < len(queue):
                queue.insert(each[1], each)
        return queue

    '''
    665. Non-decreasing Array (Easy)
    给你一个长度为 n 的整数数组，请你判断在 最多 改变 1 个元素的情况下，该数组能否变成一个非递减数列。
    我们是这样定义一个非递减数列的： 对于数组中任意的 i (0 <= i <= n-2)，总满足 nums[i] <= nums[i + 1]。

    贪心思想：参考122的思想，用一个diff列表来表示差分，如果该位置满足非递减则为0，不满足为-1，讨论diff的值即可
    '''
    def checkPossibility(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        diff = []
        for i in range(1, len(nums)):
            diff.append(0 if nums[i] - nums[i - 1] >= 0 else -1)
        # 若和为0，表明直接满足非递减
        if sum(diff) == 0:
            return True
        # 若和小于-1，则至少两处不满足，没得救
        elif sum(diff) < -1:
            return False
        # 若只有一处，分是否在左右边界谈论
        else:
            index = diff.index(-1)
            if index + 2 == len(nums) or index == 0:
                return True
            else:
                # 注意，若把数组表示为折线图，则index是折线的山峰处，index+1是山谷处，所以下面前者考虑能否“填谷”，后者考虑能否“移山”。
                return nums[index + 2] >= nums[index] or nums[index + 1] >= nums[index - 1]
    