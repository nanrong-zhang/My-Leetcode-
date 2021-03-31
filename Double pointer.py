'''
    双指针主要用于遍历数组，两个指针指向不同的元素，从而协同完成任务。也可以延伸到多个数组的多个指针。
    若两个指针指向同一数组，遍历方向相同且不会相交，则也称为滑动窗口（两个指针包围的区域即为当前的窗口），经常用于区间搜索。
    若两个指针指向同一数组，但是遍历方向相反，则可以用来进行搜索，待搜索的数组往往是排好序的。
    这一单元要注意链表的使用方式：题142

'''
class Solution(object):
    '''
    167. Two Sum II - Input array is sorted (Easy)
    给定一个已按照 升序排列  的整数数组 numbers ，请你从数组中找出两个数满足相加之和等于目标数 target 。
    函数应该以长度为 2 的整数数组的形式返回这两个数的下标值。numbers 的下标 从 1 开始计数 ，所以答案数组应当满足 1 <= answer[0] < answer[1] <= numbers.length 。
    你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。

    思想：用两个分别指向头尾的指针来遍历数组，因为数组已经排好序了，故根据target逐渐滑动指针即可
    '''
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        pointer1 = 1
        pointer2 = len(numbers)
        #双指针遍历，如果两个指针指向元素的和小于给定值，把左边的指针右移一位，使和增加一点。反之若大于给定值，把右边的指针左移一位，使得当前的和减少一点，直到等于给定值
        while True :
            if numbers[pointer1-1]+numbers[pointer2-1] < target :
                pointer1 +=1
            elif numbers[pointer1-1]+numbers[pointer2-1] > target :
                pointer2 -=1
            else :
                return [pointer1,pointer2]

    '''
    88. Merge Sorted Array (Easy)
    给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。
    初始化 nums1 和 nums2 的元素数量分别为 m 和 n 。你可以假设 nums1 的空间大小等于 m + n，这样它就有足够的空间保存来自 nums2 的元素。

    思想：同样对于排好序的数组，用两个分别指向两个数组末尾的指针来遍历数组，但是注意由于数组一末尾被添了零，所以指向的初始位置应该是m-1，然后维护一个指向数组一最后位置的指针
    反向遍历比较并赋值，还要注意另一个是结束条件是第二个数组全都插入进去为止
    '''
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        pointer1 = m - 1
        pointer2 = n - 1
        pos = m + n - 1
        while pointer1 >= 0 and pointer2 >= 0:
            if nums1[pointer1] < nums2[pointer2]:
                nums1[pos] = nums2[pointer2]
                pointer2 -= 1
            else:
                nums1[pos] = nums1[pointer1]
                pointer1 -= 1
            pos -= 1
        while pointer2 >= 0:
            nums1[pos] = nums2[pointer2]
            pointer2 -= 1
            pos -= 1
        return nums1

    '''
    142. Linked List Cycle II (Medium)
    给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
    为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 
    如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。
    说明：不允许修改给定的链表。
    进阶：
        你是否可以使用 O(1) 空间解决此题？
        
    思路：对于链表找环路的问题，有一个通用的解法——快慢指针（Floyd 判圈法，见https://en.wikipedia.org/wiki/Cycle_detection）。给定两个指针，
    分别命名为 slow 和 fast，起始位置在链表的开头。每次 fast 前进两步，slow 前进一步。如果 fast
    可以走到尽头，那么说明没有环路；如果 fast 可以无限走下去，那么说明一定有环路，且一定存
    在一个时刻 slow 和 fast 相遇。当 slow 和 fast 第一次相遇时，我们将 fast 重新移动到链表开头，并
    让 slow 和 fast 每次都前进一步。当 slow 和 fast 第二次相遇时，相遇的节点即为环路的开始点。
    '''
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        fast ,slow = head ,head
        while True :
            if not (fast and fast.next) :
                return
            fast ,slow = fast.next.next ,slow.next
            if fast == slow :
                fast = head
                break
        while True:
            if fast == slow :
                break
            fast, slow = fast.next, slow.next
        return fast

    '''
    76. Minimum Window Substring (Hard)  
    给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
    注意：如果 s 中存在这样的子串，我们保证它是唯一的答案。

    思路：
    用i,j表示滑动窗口的左边界和右边界，通过改变i,j来扩展和收缩滑动窗口，可以想象成一个窗口在字符串上游走，当这个窗口包含的元素满足条件，
    即包含字符串T的所有元素，记录下这个滑动窗口的长度j-i+1，这些长度中的最小值就是要求的结果。
    步骤一
        不断增加j使滑动窗口增大，直到窗口包含了T的所有元素
    步骤二
        不断增加i使滑动窗口缩小，因为是要求最小字串，所以将不必要的元素排除在外，使长度减小，直到碰到一个必须包含的元素，记录此时滑动窗口的长度，并保存最小值
    步骤三
        让i再增加一个位置，这个时候滑动窗口肯定不满足条件了，那么继续从步骤一开始执行，寻找新的满足条件的滑动窗口，如此反复，直到j超出了字符串S范围。
    如何判断滑动窗口包含了T的所有元素？
        利用两个字典，一个通过遍历t来统计各个字符所需要的数量，另一个保存当前窗口中各个所需要字符已经储存的数量，并维护一个count来统计还需要多少个字符
    
    '''
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        #创建两个字典，利用两个字典，一个通过遍历t来统计各个字符所需要的数量，用t的长度初始化count，用s长度加一初始化最短长度
        # 因为要求如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符，所以用一个空字符串初始化当前最短字符，
        need_dict=collections.defaultdict(int)
        own_dict=collections.defaultdict(int)
        count = len(t)
        for each in t:
            need_dict[each] +=1
        minlen = len(s)+1
        curstr = ''
        head , end = 0 , 0
        #开始遍历，先移动右指针，判断元素是否是需要的，若需要则更新第二个字典和count
        while end < len(s):
            if s[end] not in need_dict :
                end +=1
                continue
            #注意，只有在相应字符已经拥有的数量小于需要的时，才更新count
            if own_dict[s[end]] < need_dict[s[end]]:
                count -= 1
            #无论如何已拥有字典和右指针都要更新
            own_dict[s[end]]+=1
            end +=1
            #如果窗口已经满足要求 ，开始移动左指针，原理与上面的基本一致
            while count == 0 :
                #如果当前窗口更短，更新最短字符串和最短长度
                if minlen > end - head :
                    curstr = s[head:end]
                    minlen = end - head
                if s[head] not in need_dict :
                    head +=1
                    continue
                if own_dict[s[head]] == need_dict[s[head]]:
                    count += 1
                own_dict[s[head]]-=1
                head +=1

        return curstr
    '''
    633. Sum of Square Numbers (Easy)
    给定一个非负整数 c ，你要判断是否存在两个整数 a 和 b，使得 a2 + b2 = c 。
    
    思路：和上面的167题类似，建立一头一尾两个指针逐渐缩小范围来查找，由于没有给定列表，可以直接用自然数，根据题目性质最大值是c开根号后取整
    '''
    def judgeSquareSum(self, c):
        """
        :type c: int
        :rtype: bool
        """
        left, right = 0, int(math.sqrt(c))
        while left <= right:
            if left ** 2 + right ** 2 < c:
                left += 1
            elif left ** 2 + right ** 2 > c:
                right -= 1
            else:
                return True
        return False

    '''
    680. Valid Palindrome II (Easy)
    给定一个非空字符串 s，最多删除一个字符。判断是否能成为回文字符串。
    
    思路：还是一头一尾两个指针逐渐收缩，因为最多只能删除一个元素，因此判断删左边指针指向的和删右边指针指向的两种方式下结果是否还是回文数，利用lambda简化判断回文数过程
    '''
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        is_palindrome = lambda x: x == x[::-1]
        head, end = 0, len(s) - 1
        while head <= end:
            if s[head] != s[end]:
                return is_palindrome(s[head + 1: end + 1]) or is_palindrome(s[head: end])
            head += 1
            end -= 1
        return True

    '''
    524. Longest Word in Dictionary through Deleting (Medium)
    给定一个字符串和一个字符串字典，找到字典里面最长的字符串，该字符串可以通过删除给定字符串的某些字符来得到。
    如果答案不止一个，返回长度最长且字典顺序最小的字符串。如果答案不存在，则返回空字符串。

    思路：注意，给我们的是一个列表！而最长答案不止一个时是要求返回字典顺序最小的，所以需要先对列表排序，对字符串列表的排序本质和字典一样，都是按位比较大小
    如 'abc' > 'abe' > 'bbc' 
    排序后，遍历每一个列表元素，对字符串的每个字符在s中从左到右查找位置，若不存在则换下一个元素，若存在则通过其位置标签将s前面的部分切割掉（贪心思想），继续下一个字符
    最后如果每个字符都满足要求，即s可以通过删除某些字符来得到该字符串，则记录长度，最后通过长度对比确定是否更新最长符合条件字符串的位置和长度
    '''
    def findLongestWord(self, s, dictionary):
        """
        :type s: str
        :type dictionary: List[str]
        :rtype: str
        """
        maxlen = 0
        maxindex = -1
        #先对列表排序使其中字符串元素按字典的规则排序
        dictionary.sort()
        #遍历字典
        for i in range(len(dictionary)):
            curlen = 0
            tmp_s = s
            #遍历字符串元素
            for j in range(len(dictionary[i])):
                if tmp_s.find(dictionary[i][j]) == -1:
                    break
                else:
                    tmp_s = tmp_s[tmp_s.find(dictionary[i][j]) + 1:]
                    #如果已经到最后位置，说明该元素符合要求，记录其长度
                    if j == len(dictionary[i]) - 1:
                        curlen = j + 1
            #通过长度对比确定是否更新最长符合条件字符串的位置和长度
            if curlen > maxlen:
                maxlen = curlen
                maxindex = i
        if maxindex == -1:
            return ''
        else:
            return dictionary[maxindex]