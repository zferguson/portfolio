import string

class Solution:
    def decodeMessage(self, key: str, message: str) -> str:
        self.store = {}

        key = "".join(key.split())
        new_key = ""
        for char in key:
            if char not in new_key:
                new_key += char
        
        for i, char in enumerate(new_key):
            if char not in self.store.keys():
                self.store[char] = string.ascii_lowercase[i]

        output = ""

        for char in message:
            if char == " ":
                output += " "
            else:
                output += self.store[char]
        
        return output
