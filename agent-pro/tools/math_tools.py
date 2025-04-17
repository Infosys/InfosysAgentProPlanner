''' 
Copyright 2025 Infosys Ltd.
 
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all copies 
or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE 
AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

from typing import Dict, Optional, Any

async def multiply_numbers(num1: Any, num2: Any, **kwargs) -> Any:
    """Multiplies two numbers."""
    try:
        result = float(num1) * float(num2)
        print(f"  [Tool] Multiplying {num1} * {num2}")
        return result
    except ValueError:
        error_message = f"Error: Could not convert '{num1}' or '{num2}' to numbers."
        print(error_message)
        return error_message

async def divide_numbers(num1: Any, num2: Any, **kwargs) -> Any:
    """Divides two numbers."""
    try:
        result = float(num1) / float(num2)
        print(f"  [Tool] Dividing {num1} by {num2}")
        return result
    except ZeroDivisionError:
        error_message = "Error: Cannot divide by zero."
        print(error_message)
        return error_message
    except ValueError:
        error_message = f"Error: Could not convert '{num1}' or '{num2}' to numbers."
        print(error_message)
        return error_message

async def calculate_sum(num1: Any, num2: Any, **kwargs) -> Any:
    """Calculates sum of two numbers."""
    try:
        result = float(num1) + float(num2)
        print(f"  [Tool] Calculating Sum {num1} + {num2}")
        return result
    except ValueError:
        error_message = f"Error: Could not convert '{num1}' or '{num2}' to numbers."
        print(error_message)
        return error_message