from math import sqrt, floor

# LeetCode 69. Sqrt(x)
# https://leetcode.com/problems/sqrtx/
class Solution:
    def mySqrt(self, x: int) -> int:
        return floor(sqrt(x))