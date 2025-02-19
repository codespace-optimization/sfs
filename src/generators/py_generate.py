from .model import ModelBase, message_to_str, Message
# from .generator_types import Generator
from .generator import GenericChatGenerator, GeneratorConfig
from .generator_utils import generic_generate_func_impl, generic_generate_internal_tests, generic_generate_self_reflection, generate_with_accumulated_context, generic_strategist_generate_ideas, generic_strategist_generate_strategy
# from searchlightimprove.evolver_dependencies import StrategistEvolverDependency, TreeEvolverDependency, SynthesisEvolverDependency

from typing import Optional, List, Union
import ast
import re
from .parse import parse_code_block, add_code_block

def parse_updated_insights(text: str) -> str:
    # Split the text by lines
    lines = text.split('\n')
    
    # Initialize variables
    in_updated_insights = False
    insights_content = []

    # Iterate through each line to find and collect the contents of *Updated Insights*
    for line in lines:
        # Check if the line indicates the start of *Updated Insights*
        if 'Updated Insights' in line:
            in_updated_insights = True
            continue
        
        # If we are in the *Updated Insights* section and the line is not empty, collect the line
        if in_updated_insights:
            if line.strip() == '':
                break
            insights_content.append(line.strip())
    
    # Join the collected lines with newlines to form the final content
    return '\n'.join(insights_content)

USE_PYTHON_CODEBLOCK_INSTRUCTION = "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
PY_REFLEXION_FEW_SHOT_ADD = [Message(role="user", content='''[prompt]:
Write a function that takes two integers as input and returns the sum of the two integers.
                      
[previous impl]:
```python
def add(a: int, b: int) -> int:
    """
    Given integers a and b, return the total value of a and b.
    """
    return a - b
```

[unit test results from previous impl]:
Tested passed:

Tests failed:
assert add(1, 2) == 3 # output: -1
assert add(1, 2) == 4 # output: -1

[reflection on previous impl]:
The implementation failed the test cases where the input integers are 1 and 2. The issue arises because the code does not add the two integers together, but instead subtracts the second integer from the first. To fix this issue, we should change the operator from `-` to `+` in the return statement. This will ensure that the function returns the correct output for the given input.

[improved impl]:                      
'''), 
Message(role="assistant", content='''```python
def add(a: int, b: int) -> int:
    """
    Given integers a and b, return the total value of a and b.
    """
    return a + b
```'''),]

PY_REFLEXION_FEW_SHOT = [Message(role="user", 
                                 content='''[prompt]:
Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.
You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces `' '` when necessary so that each line has exactly maxWidth characters.
Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
For the last line of text, it should be left justified and no extra space is inserted between words.
Note:
A word is defined as a character sequence consisting of non-space characters only.
Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
The input array `words` contains at least one word.

[previous impl]:
```python
from typing import *
def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    """
    Given an array of words and a width maxWidth, formats the text such that each line has exactly maxWidth characters and is fully (left and right) justified.
    """
    res = []
    cur_line = []
    cur_len = 0

    for word in words:
        if cur_len + len(word) + len(cur_line) > maxWidth:
            if len(cur_line) == 1:
                res.append(cur_line[0] + ' ' * (maxWidth - cur_len))
            else:
                spaces = maxWidth - cur_len
                space_between = spaces // (len(cur_line) - 1)
                extra_spaces = spaces % (len(cur_line) - 1)
                line = ''
                for i, w in enumerate(cur_line[:-1]):
                    line += w + ' ' * (space_between + (i < extra_spaces))
                line += cur_line[-1]
                res.append(line)
            cur_line = []
            cur_len = 0
        cur_line.append(word)
        cur_len += len(word)

    last_line = ' '.join(cur_line)
    last_line += ' ' * (maxWidth - len(last_line))
    res.append(last_line)

    return res
```

[unit test results from previous impl]:
Tested passed:

Tests failed:
assert fullJustify([], 10) == [] # output: ['          ']
assert fullJustify([], 0) == [] # output: ['']

[reflection on previous impl]:
The implementation failed the test cases where the input list of words is empty. The issue arises because the code does not handle the case where there are no words to process. As a result, it still appends a line with spaces to the result list, even when there are no words. To fix this issue, we should add a condition at the beginning of the function to check if the input list is empty, and return an empty list if it is. This will ensure that the function returns the correct output for empty input lists.

[improved impl]:
'''),
Message(role="assistant", content='''```python
from typing import *
def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    """
    Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.
    You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces `' '` when necessary so that each line has exactly maxWidth characters.
    Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
    For the last line of text, it should be left justified and no extra space is inserted between words.
    Note:
    A word is defined as a character sequence consisting of non-space characters only.
    Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
    The input array `words` contains at least one word.
    """
    if not words:
        return []

    res = []
    cur_line = []
    cur_len = 0

    for word in words:
        if cur_len + len(word) + len(cur_line) > maxWidth:
            if len(cur_line) == 1:
                res.append(cur_line[0] + ' ' * (maxWidth - cur_len))
            else:
                spaces = maxWidth - cur_len
                space_between = spaces // (len(cur_line) - 1)
                extra_spaces = spaces % (len(cur_line) - 1)
                line = ''
                for i, w in enumerate(cur_line[:-1]):
                    line += w + ' ' * (space_between + (i < extra_spaces))
                line += cur_line[-1]
                res.append(line)
            cur_line = []
            cur_len = 0
        cur_line.append(word)
        cur_len += len(word)

    last_line = ' '.join(cur_line)
    last_line += ' ' * (maxWidth - len(last_line))
    res.append(last_line)

    return res
```'''),
]

PY_SELF_REFLECTION_CHAT_INSTRUCTION = "You are a Python programming assistant. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation."

PY_SELF_REFLECTION_FEW_SHOT =[Message(role="user", 
                                      content="""[prompt]:
Write a function that finds the longest subarray with a sum less than or equal to a given limit.

[function impl]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 >= max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```

[unit test results]:
Tests passing:
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 8) == [1, 2, 3]
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 15) == [1, 2, 3, 4, 5]
assert longest_subarray_with_sum_limit([1, -1, 2, -2, 3, -3], 2) == [1, -1, 2, -2, 3]
assert longest_subarray_with_sum_limit([], 10) == []
assert longest_subarray_with_sum_limit([], 0) == []
assert longest_subarray_with_sum_limit([], -5) == []  

Tests failing:
assert longest_subarray_with_sum_limit([5, 6, 7, 8, 9], 4) == [] # output: [5]

[self-reflection]:
"""),
Message(role="assistant",
        content="""
The implementation failed the where no subarray fulfills the condition. The issue in the implementation is due to the use of >= instead of > in the condition to update the result. Because of this, it returns a subarray even when the sum is greater than the target, as it still updates the result when the current subarray length is equal to the previous longest subarray length. To overcome this error, we should change the condition to only update the result when the current subarray length is strictly greater than the previous longest subarray length. This can be done by replacing >= with > in the condition.
"""),
Message(role="user",
        content="""[prompt]:
Write a function that finds the longest subarray with a sum less than or equal to a given limit.

[function impl]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while current_sum + nums[right] <= target:
        current_sum += nums[right]
        right += 1
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 > max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```

[unit test results]:
Tests passing:
assert longest_subarray_with_sum_limit([], 10) == []
assert longest_subarray_with_sum_limit([], 0) == []
assert longest_subarray_with_sum_limit([], -5) == []

Tests failing:
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 8) == [1, 2, 3] # output: list index out of range
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 15) == [1, 2, 3, 4, 5] # output: list index out of range
assert longest_subarray_with_sum_limit([5, 6, 7, 8, 9], 4) == [] # output: list index out of range
assert longest_subarray_with_sum_limit([1, -1, 2, -2, 3, -3], 2) == [1, -1, 2, -2, 3] # output: list index out of range

[self-reflection]:
"""),
Message(role="assistant",
        content="""The implementation failed 4 out of the 7 test cases due to an IndexError. The issue stems from the while loop while current_sum + nums[right] <= target:, which directly accesses nums[right] without checking if right is within the bounds of the list. This results in a runtime error when right goes beyond the list length. To overcome this error, we need to add a bounds check for the right variable in the mentioned while loop. We can modify the loop condition to while right < len(nums) and current_sum + nums[right] <= target:. This change will ensure that we only access elements within the bounds of the list, thus avoiding the IndexError.""",)
        ]

PY_TEST_GENERATION_FEW_SHOT = [Message(role="user", 
                                      content="""[object name]:
add3Numbers

[object description]:
def add3Numbers(x, y, z):
    \"\"\" Add three numbers together.
    This function takes three numbers as input and returns the sum of the three numbers.
    \"\"\"

[unit tests]:
"""),
Message(role="assistant", 
        content="""```python
assert add3Numbers(1, 2, 3) == 6, "Addition of positive numbers"
assert add3Numbers(-1, 2, 3) == 4, "Addition of negative and positive numbers"
assert add3Numbers(1.0, 2.0, 3.0) == 6.0, "Addition of floating-point numbers"
assert add3Numbers(-3, -2, -1) == -6, "Addition of negative numbers"
assert add3Numbers(0, 0, 0) == 0, "Addition of zeros"
```""")
]

PY_TEST_GENERATION_CHAT_INSTRUCTION = """You are an AI coding assistant that can write unique, diverse, and comprehensive unit tests for python objects given the description of the object. The format of test cases should be:
```python
assert function_name(input_1) == expected_output_1, "Test case 1 description"
assert function_name(input_2) == expected_output_2, "Test case 2 description"
```
DO NOT use pytest or unittest frameworks for this task. Stick to small inputs that you can easily verify the output for."""

PY_COT_TEST_GENERATION_CHAT_INSTRUCTION = """You are an software engineer capable of writing unique, diverse, and comprehensive unit tests for python objects given the description of the object. The format of test cases should be:
```python
assert function_name(input_1) == expected_output_1, "Test case 1 description"
```
You should work through the function step by step from input to output for each unit test to ensure correctness. Stick to small inputs that you can easily verify the output for."""

PY_COT_TEST_GENERATION_FEW_SHOT = [Message(role="user", 
                                      content="""[object name]:
football_handshakes

[object description]:
def football_handshakes(n, m):
    ''' 
    Two teams of n and m players respectively just finished a football game and decided to shake hands. Each team lines up in a row and players from the first team shake hands with players from the second team. Return the total number of handshakes that occur.
    '''

[3 unit tests]:
"""),
Message(role="assistant", 
        content="""*Test case 1* 
Input = 0,0
No handshakes as there are no players
Output = 0
```python
assert football_handshakes(0, 0) == 0, "No handshakes"
```

*Test case 2* 
Input = 2, 1
Player 1 from team 1 shakes hands with Player 1 from team 2
Player 2 from team 1 shakes hands with Player 1 from team 2
Total handshakes = 2
Output = 2
```python
assert football_handshakes(2, 1) == 2, "Handshakes between two teams"
```

*Test case 3* 
Input = 2, 3
Player 1 from team 1 shakes hands with Player 1 from team 2
Player 1 from team 1 shakes hands with Player 2 from team 2
Player 1 from team 1 shakes hands with Player 3 from team 2
Player 2 from team 1 shakes hands with Player 1 from team 2
Player 2 from team 1 shakes hands with Player 2 from team 2
Player 2 from team 1 shakes hands with Player 3 from team 2
Total handshakes = 6
Output = 6
```python
assert football_handshakes(2, 3) == 6, "Handshakes between two larger teams"
```
""")
]

PY_TEST_REFINE_FEW_SHOT = [Message(role="user", 
                                      content="""[object name]:
football_handshakes

[object description]:
def football_handshakes(n, m):
    ''' 
    Two teams of n and m players respectively just finished a football game and decided to shake hands. Each team lines up in a row and players from the first team shake hands with players from the second team. Return the total number of handshakes that occur.
    '''

[proposed test]:
assert football_handshakes(2, 2) == 2, "Handshakes between two teams"

[analysis and improved test]:
"""),
Message(role="assistant", 
        content="""*Analysis*:
Input = 2, 2
Player 1 from team 1 shakes hands with Player 1 from team 2
Player 1 from team 1 shakes hands with Player 2 from team 2
Player 2 from team 1 shakes hands with Player 1 from team 2
Player 2 from team 1 shakes hands with Player 2 from team 2
Total handshakes = 4
Output = 4
The original test case is incorrect as it does not account for all possible handshakes between the two teams.

*Improved test*:
```python
assert football_handshakes(2, 2) == 4, "Handshakes between two teams"
```""")
]

PY_TEST_REFINE_CHAT_INSTRUCTION = """You are an AI coding assistant that can analyze whether a given unit test is correct or not and improve the test cases for python objects given the description of the object. Provide both the analysis of the original unit test and the improved unit tests."""

# PY_COT_TEST_GENERATION_FEW_SHOT = [Message(role="user", 
#                                       content="""[object name]:
# add3Numbers

# [object description]:
# def add3Numbers(x, y, z):
#     ''' Add three numbers together.
#     This function takes three numbers as input and returns the sum of the three numbers.
#     '''

# [3 unit tests]:
# """),
# Message(role="assistant", 
#         content="""*Test case 1* Addition of positive numbers
# Input = 1,2,3
# 1+2 = 3 # add first two numbers together
# 3+3 = 6 # add sum of first two numbers with third number
# Hence, the sum of the three numbers is 6
# Output = 6
# ```python
# assert add3Numbers(1, 2, 3) == 6, "Addition of positive numbers"
# ```

# *Test case 2* Addition of negative and positive numbers
# Input = -1, 2, 3
# -1, 2= 1 # add first two numbers together
# 1+3 = 4 # add sum of first two numbers with third number
# Hence, the sum of the three numbers is 4
# Output = 4
# ```python
# assert add3Numbers(-1, 2, 3) == 4, "Addition of negative and positive numbers"
# ```

# *Test case 3* Addition of floating-point numbers
# Input = 1.0, 2.0, 3.0
# 1.0+2.0 = 3.0 # add first two numbers together
# 3.0+3.0 = 6.0 # add sum of first two numbers with third number
# Hence, the sum of the three numbers is 6.0
# Output = 6.0
# ```python
# assert add3Numbers(1.0, 2.0, 3.0) == 6.0, "Addition of floating-point numbers"
# ```
# """)
# ]

PY_STRATEGIST_IDEA_GENERATION_FEW_SHOT =[Message(role="user", 
                                      content="""[prompt]:
def gcd(a: int, b: int) -> int:
    '''
    Given two integers a and b, return the greatest common divisor of a and b.
    '''

[previous impl]:
```python
def gcd(a: int, b: int) -> int:
    '''
    Given two integers a and b, return the greatest common divisor of a and b.
    '''
    if a >= b:
        return b if a % b == 0 else 1
    else:
        return a if b % a == 0 else 1
```

[unit test results for previous impl]:
Tested passed:
assert gcd(10, 5) == 5 # output: 5
assert gcd(10, 3) == 1 # output: 1

Tests failed:
assert gcd(6, 8) == 2 # output: 1

What are some reflections you can draw and what are 2 ways in which you can fix the previous impl so that it passes all the tests? Be as specific and concrete as possible, mentioning specific code blocks or helper functions that you will modify. Remember to number your ideas as Idea 1:, Idea 2:, ... etc.
[reflection and 2 improvement ideas]:
"""),
Message(role="assistant",
        content="""Reflection: The previous implementation does not correctly calculate the greatest common divisor (GCD) of two numbers when neither divides the other perfectly (phenomenon), as seen with inputs 6 and 8 (evidence). It only checks if one number divides the other, returning one of the numbers or 1 in other cases, which is not the correct approach for finding the GCD (cause).

Idea 1: Use the Euclidean Algorithm Iteratively. This method keeps finding remainders until one of the numbers becomes zero, at which point the other number is the GCD.
Idea 2: Implement a Recursive GCD Function. This function will call itself with the second number and the remainder of the first number divided by the second number until the second number becomes zero, at which point the first number is the GCD.
""")
]

PY_SCOUT_IDEA_GENERATION_FEW_SHOT =[Message(role="user", 
                                      content="""[prompt]:
def gcd(a: int, b: int) -> int:
    '''
    Given two integers a and b, return the greatest common divisor of a and b.
    '''

[previous impl]:
```python
def gcd(a: int, b: int) -> int:
    '''
    Given two integers a and b, return the greatest common divisor of a and b.
    '''
    if a >= b:
        return b if a % b == 0 else 1
    else:
        return a if b % a == 0 else 1
```

[unit test results for previous impl]:
Tested passed:
assert gcd(10, 5) == 5 # output: 5
assert gcd(10, 3) == 1 # output: 1

Tests failed:
assert gcd(6, 8) == 2 # output: 1


[general insights]:
- Consider using iterative or recursive methods to find the GCD.

What are some reflections you can draw and what are 2 ways in which you can fix the previous impl so that it passes all the tests? Be as specific and concrete as possible, mentioning specific code blocks or helper functions that you will modify. Remember to number your ideas as Idea 1:, Idea 2:, ... etc.
[reflection and 2 improvement ideas]:
"""),
Message(role="assistant",
        content="""Reflection: The previous implementation does not correctly calculate the greatest common divisor (GCD) of two numbers when neither divides the other perfectly (phenomenon), as seen with inputs 6 and 8 (evidence). It only checks if one number divides the other, returning one of the numbers or 1 in other cases, which is not the correct approach for finding the GCD (cause). The general insight provided is to consider using iterative or recursive methods to find the GCD (insights).

Idea 1: Use the Euclidean Algorithm Iteratively. This method keeps finding remainders until one of the numbers becomes zero, at which point the other number is the GCD.
Idea 2: Implement a Recursive GCD Function. This function will call itself with the second number and the remainder of the first number divided by the second number until the second number becomes zero, at which point the first number is the GCD.
""")
]

# PY_STRATEGIST_IDEA_GENERATION_INSTRUCTION = "You are a Python writing assistant. You will be given a function implementation and a series of unit tests results. Your goal is to reflect on the implementation and test results and propose a few ways to modify the function so that it achieves its intended purposes and passes the tests. You will need this as a hint when you try again later. Only provide the improvement ideas, not the modified function implementation.\n\n-----"

PY_STRATEGIST_IDEA_GENERATION_INSTRUCTION = "You are a Python writing assistant. You will be given a function implementation, a series of unit test results. Explain why the implementation failed the unit tests and propose a few ways to modify the function so that it meets the requirements of the prompt and passes the unit tests. If you think the unit tests are incorrect, you can also note this. Only provide the improvement ideas that *modify the function*, not the modified function implementation.\n\n-----"

PY_SCOUT_IDEA_GENERATION_INSTRUCTION = "You are a Python writing assistant. You will be given a function implementation, a series of unit test results, and some general insights you learned previously. Explain why the implementation failed the unit tests and propose a few ways to modify the function so that it meets the requirements of the prompt and passes the unit tests. If you think the unit tests are incorrect, you can also note this. Only provide the improvement ideas that *modify the function*, not the modified function implementation.\n\n-----"

PY_STRATEGIST_GENERATE_STRATEGY_INSTRUCTION = "You are an AI Python assistant. You will be given your past function implementation, a series of unit tests and test results, and a improvement idea to change the implementation appropriately. Your job is to apply the improvement idea into your previous function implementation."

PY_STRATEGIST_GENERATE_STRATEGY_FEW_SHOT = [Message(role="user", 
                                      content="""[prompt]:
def add(a: int, b: int) -> int:
    '''
    Given integers a and b, return the total value of a and b.
    '''

[previous impl]:
```python
def add(a: int, b: int) -> int:
    '''
    Given integers a and b, return the total value of a and b.
    '''
    return a - b
```

[unit test results from previous impl]:
Tested passed:

Tests failed:
assert add(1, 2) == 3 # output: -1
assert add(2, 2) == 4 # output: 0

[improvement idea]:
Change the operator from `-` to `+` in the return statement.

Incorporate the idea above into the previous impl as best as you can.

[improved impl]:
"""), 
Message(role="assistant",
        content="""
```python
def add(a: int, b: int) -> int:
    '''
    Given integers a and b, return the total value of a and b.
    '''
    return a + b
```
""")
]

PY_SIMPLE_GENERATE_FEW_SHOT = [Message(role="user",
                                      content='''[object name]:
add

[object description]:
Write a function that, when given integers a and b, returns the sum of of a and b.

[object impl]:'''
,),
Message(role="assistant",
        content='''```python
def add(a: int, b: int) -> int:
    """
    Given integers a and b, returns the sum of a and b.
    """
    return a + b
```''',)
]

PY_SUMMARY_UPDATE_INSTRUCTION = "You are a software programming assistant. You will be given a past code implementation, feedback on its performance, an idea for changing the implementation, a new code implementation, feedback on the new implementation's performance, and insights you previously learned. Your task is to update your previous insights based on what you learned from applying the new idea to the old code implementation. The updated insights should be general, focused on why the improvement idea did or did not work, and not specific to that particular code implementation. Your insights will be used when suggesting future improvement ideas."

PY_SUMMARY_UPDATE_FEW_SHOT = [Message(role="user", 
                                      content=f"""[prompt]:
Write a python function to remove first and last occurrence of a given character from the string.

[previous implementation]:
```python
def remove_occurances(s: str, c: str) -> str:
    '''
    Given a string s and a character c, remove the first and last occurrence of c from s.
    '''
    s = s.replace(c, "", 1)

    return s
```

[previous implementation feedback]:
Test cases passed:
assert remove_occurances("PHP","H") == "PP", "Case where only one occurrence of character" # output: "PP"

Test cases failed:
assert remove_Occ("hello","l") == "heo", "Case with two occurrences of character" # output: "helo"
assert remove_Occ("abacda","a") == "bacd", "Case with three occurrences of character" # output: "bacda"

[improvement idea]:
Reverse the string and remove the first occurrence of the character, which will be the last occurrence in the original string.

[new implementation]:
```python
def remove_occurances(s: str, c: str) -> str:
    '''
    Given a string s and a character c, remove the first and last occurrence of c from s.
    '''
    s = s.replace(c, "", 1)
    s = s[::-1].replace(c, "", 1)

    return s
```

[new implementation feedback]:
Test cases passed:
assert remove_occurances("PHP","H") == "PP", "Case where only one occurrence of character" # output: "PP"

Test cases failed:
assert remove_Occ("hello","l") == "heo", "Case with two occurrences of character" # output: "oeh"
assert remove_Occ("abacda","a") == "bacd", "Case with three occurrences of character" # output: "dcab"

[score improvement]:
0.0

[previous insights]:
- We need to make sure we remove both the first and last occurrence of the character from the string

Reflect on how the improvement idea changed the performance of the code implementation and update your insights accordingly. You have have a separate *Reflections* section and *Insights* section.
[updated insights]:
"""), 
Message(role="assistant",
        content="""
*Reflections*
The new implementation still fails the same test cases as the previous implementation, resulting in no score improvement (observation). The issue with this improvement idea is that while it removes the first occurrence of the character from the reversed string—which corresponds to the last occurrence in the original string—we return the reversed string as the final output (cause). This leads to the output being reversed, which is not the desired behavior. Therefore, this improvement idea is incomplete but a step in the right direction (conclusion on improvement idea). In future improvements, we should consider reversing the string back to its original order after removing the last occurrence of the character.

*Updated Insights*
Ensure that we remove both the first and last occurrences of the character from the string when suggesting improvements.
When suggesting a modification to the implementation, ensure that the modification does not change the order of the characters in the string.
""")
]

DUMMY_FUNC_SIG = "def func():"
DUMMY_FUNC_CALL = "func()"


def handle_first_line_indent(func_body: str) -> str:
    if func_body.startswith("    "):
        return func_body
    split = func_body.splitlines()
    return f"    {split[0]}\n" + "\n".join(split[1:])


def handle_entire_body_indent(func_body: str) -> str:
    split = func_body.splitlines()
    res = "\n".join(["    " + line for line in split])
    return res


def fix_turbo_response(func_body: str) -> str:
    return fix_markdown(remove_unindented_signatures(func_body))


def fix_markdown(func_body: str) -> str:
    return re.sub("`{3}", "", func_body)


def remove_unindented_signatures(code: str) -> str:
    regex = r"^def\s+\w+\s*\("

    before_signature = []
    after_signature = []
    signature_found = False

    for line in code.split("\n"):
        if re.match(regex, line):
            signature_found = True
            continue

        if signature_found:
            after_signature.append(line)
        else:
            if not line.startswith("    ") and line.strip():
                line = "    " + line
            before_signature.append(line)

    return "\n".join(before_signature + after_signature)


def py_fix_indentation(func_body: str) -> str:
    func_body = fix_turbo_response(func_body)
    """
    3 cases:
        1. good syntax
        2. first line not good
        3. entire body not good
    """
    def parse_indent_rec(f_body: str, cur_state: int) -> str:
        f_body = fix_markdown(f_body)
        if cur_state > 1:
            return f_body
        code = f'{DUMMY_FUNC_SIG}\n{f_body}\n{DUMMY_FUNC_CALL}'
        try:
            exec(code)
            return f_body
        except (IndentationError, SyntaxError):
            p_func = handle_first_line_indent if cur_state == 0 else handle_entire_body_indent
            return parse_indent_rec(p_func(func_body), cur_state + 1)
        except Exception:
            return f_body
    return parse_indent_rec(func_body, 0)


def py_is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

def parse_tests(tests: str) -> List[str]:
    return [test.strip() for test in tests.splitlines() if "assert" in test]

def parse_improvement_ideas(string: str, num_ideas: int) -> list[str]:
    '''
    Find patterns of "Idea X: <idea here>", "**Idea X:** <idea here>", "*Idea X:* <idea here>", etc. in the input string and return the first num_ideas ideas.
    '''
    # Updated pattern to capture variations like "**Idea X:", "*Idea X:*", etc.
    idea_pattern = re.compile(r"\**Idea \d+:\**\s*(.+)")

    # Find all matches of the pattern in the input string
    matches = idea_pattern.findall(string)

    # Return the first num_ideas ideas from the matches
    out = matches[:num_ideas]

    # fill with blanks if not enough ideas
    if len(out) < num_ideas:
        out += ["Improve the code."] * (num_ideas - len(out))

    return out


python_config = GeneratorConfig(
    code_block_instruction=USE_PYTHON_CODEBLOCK_INSTRUCTION,
    parse_code_block=lambda x: parse_code_block(x, "python"),
    add_code_block=lambda x: add_code_block(x, "python"),
    parse_ideas=parse_improvement_ideas,
    parse_insights=parse_updated_insights,
    parse_tests=parse_tests,
    is_syntax_valid=py_is_syntax_valid,

    reflection_chat_instruction="You are an AI Python assistant. You will be given your past implementation, a series of unit tests, and a hint to change the implementation appropriately. Write your full improved implementation, including necessary imports, but nothing else.",
    reflection_few_shot=PY_REFLEXION_FEW_SHOT_ADD,

    simple_chat_instruction="You are an AI that only responds with python code, NOT ENGLISH. You will be given the name of a python function or class that you need to implement. Write your full implementation of the function or class based on the given description of it, including necessary imports, but nothing else.",
    simple_generate_few_shot=PY_SIMPLE_GENERATE_FEW_SHOT,

    self_reflection_few_shot=PY_SELF_REFLECTION_FEW_SHOT,
    self_reflection_chat_instruction=PY_SELF_REFLECTION_CHAT_INSTRUCTION,
    
    test_generation_few_shot=PY_TEST_GENERATION_FEW_SHOT,
    test_generation_chat_instruction=PY_TEST_GENERATION_CHAT_INSTRUCTION,
    
    idea_generation_instruction=PY_STRATEGIST_IDEA_GENERATION_INSTRUCTION,
    idea_generation_few_shot=PY_STRATEGIST_IDEA_GENERATION_FEW_SHOT,
    
    strategy_generation_instruction=PY_STRATEGIST_GENERATE_STRATEGY_INSTRUCTION,strategy_generation_few_shot=PY_STRATEGIST_GENERATE_STRATEGY_FEW_SHOT,

    summary_update_instruction=PY_SUMMARY_UPDATE_INSTRUCTION,
    summary_update_few_shot=PY_SUMMARY_UPDATE_FEW_SHOT,

    test_generation_few_shot_cot=PY_COT_TEST_GENERATION_FEW_SHOT,
    test_generation_chat_instruction_cot=PY_COT_TEST_GENERATION_CHAT_INSTRUCTION,

    test_refine_few_shot=PY_TEST_REFINE_FEW_SHOT,
    test_refine_chat_instruction=PY_TEST_REFINE_CHAT_INSTRUCTION,

    scout_idea_generation_few_shot=PY_SCOUT_IDEA_GENERATION_FEW_SHOT,
    scout_idea_generation_instruction=PY_SCOUT_IDEA_GENERATION_INSTRUCTION,
)

class PyGenerator(GenericChatGenerator):
    
    def __init__(self,):
        super().__init__(python_config)
    
    # def func_impl(
    #     self,
    #     func_sig: str,
    #     model: ModelBase,
    #     strategy: str,
    #     entry_point: str,
    #     prev_func_impl: Optional[str] = None,
    #     feedback: Optional[str] = None,
    #     self_reflection: Optional[str] = None,
    #     num_comps: int = 1,
    #     temperature: float = 0.8,
    #     preamble: str = '',
    #     acc_feedback: Optional[str] = None,
    #     acc_reflection: Optional[str] = None,
    # ) -> list[Optional[str]]:
    #     if strategy == "mcts":
    #         raise NotImplementedError("MCTS strategy is not supported for Python.")
    #     else:
    #         return self.generate_func_impl(func_sig=func_sig, strategy=strategy, prev_func_impl=prev_func_impl, feedback=feedback, self_reflection=self_reflection, num_comps=num_comps, temperature=temperature, model=model, preamble=preamble, entry_point=entry_point)
        
    def self_reflection(self, func: str, feedback: str, model: ModelBase, temperature: float) -> str:
        return self.generate_self_reflection(func=func, feedback=feedback, model=model, temperature= temperature,)